import consola from "consola"
import { events } from "fetch-event-stream"

import type {
  AnthropicMessagesPayload,
  AnthropicResponse,
} from "~/routes/messages/anthropic-types"
import type { SubagentMarker } from "~/routes/messages/subagent-marker"

import { copilotBaseUrl, copilotHeaders } from "~/lib/api-config"
import { HTTPError } from "~/lib/error"
import { state } from "~/lib/state"
import { fetchCopilotWithRetry } from "~/services/copilot/request"

export type MessagesStream = ReturnType<typeof events>
export type CreateMessagesReturn = AnthropicResponse | MessagesStream

const SUPPORTED_BETA_FEATURES = new Set([
  "advanced-tool-use-2025-11-20",
  "interleaved-thinking-2025-05-14",
])

function filterBetaHeader(header: string): string | undefined {
  const supported = header
    .split(",")
    .map((s) => s.trim())
    .filter((s) => SUPPORTED_BETA_FEATURES.has(s))
  return supported.length > 0 ? supported.join(",") : undefined
}

interface SubagentInfo {
  subagentMarker: SubagentMarker | null
  sessionId: string | undefined
}

interface CreateMessagesOptions extends SubagentInfo {
  anthropicBetaHeader?: string
  initiatorOverride?: "agent" | "user"
}

const applySubagentHeaders = (
  sessionId: string | undefined,
  isSubagent: boolean,
  headers: Record<string, string>,
): void => {
  if (isSubagent) {
    headers["x-initiator"] = "agent"
    headers["x-interaction-type"] = "conversation-subagent"
  }

  if (sessionId) {
    headers["x-interaction-id"] = sessionId
  }
}

export const createMessages = async (
  payload: AnthropicMessagesPayload,
  options: CreateMessagesOptions = {
    anthropicBetaHeader: undefined,
    initiatorOverride: undefined,
    sessionId: undefined,
    subagentMarker: null,
  },
): Promise<CreateMessagesReturn> => {
  const enableVision = payload.messages.some(
    (message) =>
      Array.isArray(message.content)
      && message.content.some((block) => block.type === "image"),
  )

  const inferredInitiator = (): "agent" | "user" => {
    const lastMessage = payload.messages.at(-1)
    if (lastMessage?.role !== "user") return "user"
    const hasUserInput =
      Array.isArray(lastMessage.content) ?
        lastMessage.content.some((block) => block.type !== "tool_result")
      : true
    return hasUserInput ? "user" : "agent"
  }

  const initiator = options.initiatorOverride ?? inferredInitiator()

  // Remove unsupported fields that Copilot API rejects
  // biome-ignore lint/performance/noDelete: cleaning up unsupported fields
  delete (payload as unknown as Record<string, unknown>).context_management

  const buildHeaders = () => {
    const headers: Record<string, string> = {
      ...copilotHeaders(state, enableVision),
      "X-Initiator": initiator,
    }

    const filteredBeta =
      options.anthropicBetaHeader ?
        filterBetaHeader(options.anthropicBetaHeader)
      : undefined
    if (filteredBeta) {
      headers["anthropic-beta"] = filteredBeta
    } else if (payload.thinking?.budget_tokens) {
      headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
    }

    applySubagentHeaders(
      options.sessionId,
      Boolean(options.subagentMarker),
      headers,
    )

    return headers
  }

  const response = await fetchCopilotWithRetry({
    url: `${copilotBaseUrl(state)}/v1/messages`,
    init: {
      method: "POST",
      body: JSON.stringify(payload),
    },
    buildHeaders,
  })

  if (!response.ok) {
    consola.error("Failed to create messages", response)
    throw new HTTPError("Failed to create messages", response)
  }

  if (payload.stream) {
    return events(response)
  }

  return (await response.json()) as AnthropicResponse
}
