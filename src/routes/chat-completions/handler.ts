import type { Context } from "hono"

import consola from "consola"
import { streamSSE, type SSEMessage } from "hono/streaming"

import { getMappedModel, getReasoningEffortForModel } from "~/lib/config"
import { createHandlerLogger } from "~/lib/logger"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import { getTokenCount } from "~/lib/tokenizer"
import { isNullish } from "~/lib/utils"
import {
  createChatCompletions,
  type ChatCompletionChunk,
  type ChatCompletionResponse,
  type ChatCompletionsPayload,
  type Message,
} from "~/services/copilot/create-chat-completions"
import {
  createResponses,
  type ResponsesPayload,
  type ResponsesResult,
  type ResponseStreamEvent,
} from "~/services/copilot/create-responses"

const logger = createHandlerLogger("chat-completions-handler")

const requiresResponsesEndpoint = (modelId: string): boolean => {
  const selectedModel = state.models?.data.find((m) => m.id === modelId)
  if (!selectedModel) return false
  if (selectedModel.supported_endpoints) {
    return (
      !selectedModel.supported_endpoints.includes("/chat/completions")
      && selectedModel.supported_endpoints.includes("/responses")
    )
  }
  return selectedModel.vendor === "OpenAI"
}

const translateMessagesToResponsesInput = (
  messages: Array<Message>,
): ResponsesPayload["input"] => {
  return messages.map((msg) => ({
    type: "message" as const,
    role: msg.role as "user" | "assistant" | "system" | "developer",
    content:
      typeof msg.content === "string" ?
        msg.content
      : (msg.content ?? []).map((part) => {
          if (part.type === "text")
            return { type: "input_text" as const, text: part.text }
          return part
        }),
  }))
}

const responsesResultToChatCompletion = (
  result: ResponsesResult,
  model: string,
): ChatCompletionResponse => {
  const content = result.output
    .filter((item) => item.type === "message")
    .flatMap((item) =>
      "content" in item && Array.isArray(item.content) ? item.content : [],
    )
    .filter((block) => block.type === "output_text")
    .map((block) => ("text" in block ? block.text : ""))
    .join("")

  return {
    id: result.id,
    object: "chat.completion",
    created: result.created_at,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        logprobs: null,
        finish_reason: result.status === "completed" ? "stop" : "length",
      },
    ],
    usage:
      result.usage ?
        {
          prompt_tokens: result.usage.input_tokens,
          completion_tokens: result.usage.output_tokens ?? 0,
          total_tokens: result.usage.total_tokens,
        }
      : undefined,
  }
}

const responsesStreamEventToChatChunk = (
  event: ResponseStreamEvent,
  model: string,
  chunkId: string,
): ChatCompletionChunk | null => {
  if (event.type === "response.output_text.delta") {
    return {
      id: chunkId,
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [
        {
          index: 0,
          delta: { content: event.delta },
          finish_reason: null,
          logprobs: null,
        },
      ],
    }
  }
  if (
    event.type === "response.completed"
    || event.type === "response.incomplete"
  ) {
    return {
      id: chunkId,
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason:
            event.type === "response.completed" ? "stop" : "length",
          logprobs: null,
        },
      ],
    }
  }
  return null
}

export async function handleCompletion(c: Context) {
  await checkRateLimit(state)

  let payload = await c.req.json<ChatCompletionsPayload>()
  consola.info(`[Request] model: ${payload.model}`)
  logger.debug("Request payload:", JSON.stringify(payload).slice(-400))

  payload.model = getMappedModel(payload.model)

  const selectedModel = state.models?.data.find(
    (model) => model.id === payload.model,
  )

  try {
    if (selectedModel) {
      const tokenCount = await getTokenCount(payload, selectedModel)
      logger.info("Current token count:", tokenCount)
    } else {
      logger.warn("No model selected, skipping token count calculation")
    }
  } catch (error) {
    logger.warn("Failed to calculate token count:", error)
  }

  if (isNullish(payload.max_tokens)) {
    payload = {
      ...payload,
      max_tokens: selectedModel?.capabilities.limits.max_output_tokens,
    }
    logger.debug("Set max_tokens to:", JSON.stringify(payload.max_tokens))
  }

  if (requiresResponsesEndpoint(payload.model)) {
    logger.debug(
      `Model ${payload.model} requires /responses endpoint, translating request`,
    )
    return handleViaResponsesEndpoint(c, payload)
  }

  const response = await createChatCompletions(payload)

  if (isNonStreaming(response)) {
    logger.debug("Non-streaming response:", JSON.stringify(response))
    return c.json(response)
  }

  logger.debug("Streaming response")
  return streamSSE(c, async (stream) => {
    for await (const chunk of response) {
      logger.debug("Streaming chunk:", JSON.stringify(chunk))
      await stream.writeSSE(chunk as SSEMessage)
    }
  })
}

const handleViaResponsesEndpoint = async (
  c: Context,
  payload: ChatCompletionsPayload,
) => {
  const lastMessage = payload.messages.at(-1)
  const isAgentCall =
    lastMessage !== undefined
    && (lastMessage.role === "assistant" || lastMessage.role === "tool")

  const responsesPayload: ResponsesPayload = {
    model: payload.model,
    input: translateMessagesToResponsesInput(payload.messages),
    tools: payload.tools?.map((t) => ({
      type: "function" as const,
      name: t.function.name,
      description: t.function.description,
      parameters: t.function.parameters,
      strict: null,
    })),
    tool_choice: payload.tool_choice as ResponsesPayload["tool_choice"],
    stream: payload.stream ?? false,
    max_output_tokens: payload.max_tokens ?? undefined,
    reasoning: { effort: getReasoningEffortForModel(payload.model) },
    service_tier: null,
  }

  const response = await createResponses(responsesPayload, {
    vision: false,
    initiator: isAgentCall ? "agent" : "user",
  })

  if (payload.stream && isAsyncIterable(response)) {
    const modelId = payload.model
    return streamSSE(c, async (stream) => {
      let chunkId = `chatcmpl-${Date.now()}`
      for await (const rawChunk of response) {
        const data = (rawChunk as { data?: string }).data
        if (!data) continue
        try {
          const event = JSON.parse(data) as ResponseStreamEvent
          if (event.type === "response.created") {
            chunkId =
              (event as { response?: { id?: string } }).response?.id ?? chunkId
          }
          const chunk = responsesStreamEventToChatChunk(event, modelId, chunkId)
          if (chunk) {
            await stream.writeSSE({ data: JSON.stringify(chunk) })
          }
        } catch {
          // skip unparseable chunks
        }
      }
      await stream.writeSSE({ data: "[DONE]" })
    })
  }

  const result = response as ResponsesResult
  return c.json(responsesResultToChatCompletion(result, payload.model))
}

const isAsyncIterable = <T>(value: unknown): value is AsyncIterable<T> =>
  Boolean(value)
  && typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === "function"

const isNonStreaming = (
  response: Awaited<ReturnType<typeof createChatCompletions>>,
): response is ChatCompletionResponse => Object.hasOwn(response, "choices")
