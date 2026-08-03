import Foundation

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let port: Int
    /// Explicit --model-id value; nil defers to the loaded model's family
    /// default (gemma-4-26b-a4b-it or qwen3.6-35b-a3b).
    public let modelIDOverride: String?
    public var modelID: String { modelIDOverride ?? "gemma-4-26b-a4b-it" }
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    public let expertCacheSlots: Int?
    public let rdadvise: String?

    public static let usage = """
    usage: TurboFieldfareServer --model <completed .gturbo directory> [options]

      --model <dir>          Required model directory.
      --port <1...65535>     Loopback port (default 8000).
      --model-id <id>        API model identifier (default derived from the
                             installed model: gemma-4-26b-a4b-it,
                             qwen3.6-35b-a3b, or qwen3.6-14b-a3b).
      --max-context <tokens> 4096, 8192, 16384, 32768, or 65536 (default 16384).
      --queue-limit <count>  Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix>
                             Prompt KV reuse mode (default single-prefix).
      --expert-cache-slots <n>  Routed-expert cache slots per layer: 8, 16,
                                24, 32, or 90. Default auto: cache the whole
                                expert pool when the model is small enough
                                (90 slots), else 16.
      --rdadvise <mode>      Expert read-ahead advice: off, default, bounded,
                             or adaptive (default off).
      --help                 Show this help.
    """

    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var port = 8000
        var modelIDOverride: String?
        var maxContext = 16_384
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .singlePrefix
        var expertCacheSlots: Int? = nil
        var rdadvise: String? = nil
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelIDOverride = value
            case "--max-context":
                guard let parsed = Int(value),
                      [4_096, 8_192, 16_384, 32_768, 65_536].contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off or single-prefix")
                }
                promptCacheMode = parsed
            case "--expert-cache-slots":
                guard let parsed = Int(value),
                      [8, 16, 24, 32, 90].contains(parsed) else {
                    throw ServerArgumentError.invalid("--expert-cache-slots is not supported")
                }
                expertCacheSlots = parsed
            case "--rdadvise":
                guard ["off", "default", "bounded", "adaptive"].contains(value.lowercased()) else {
                    throw ServerArgumentError.invalid("--rdadvise is not supported")
                }
                rdadvise = value
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        return ServerArguments(model: model,
                               port: port,
                               modelIDOverride: modelIDOverride,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode,
                               expertCacheSlots: expertCacheSlots,
                               rdadvise: rdadvise)
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
