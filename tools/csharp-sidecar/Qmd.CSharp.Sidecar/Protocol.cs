using System.Text.Json.Serialization;

namespace Qmd.CSharp.Sidecar;

public sealed class AnalysisRequest
{
    [JsonPropertyName("version")]
    public int Version { get; init; }

    [JsonPropertyName("language")]
    public string Language { get; init; } = string.Empty;

    [JsonPropertyName("filePath")]
    public string FilePath { get; init; } = string.Empty;

    [JsonPropertyName("content")]
    public string Content { get; init; } = string.Empty;

    [JsonPropertyName("features")]
    public AnalysisFeatures? Features { get; init; }
}

public sealed class AnalysisFeatures
{
    [JsonPropertyName("breakpoints")]
    public bool Breakpoints { get; init; }

    [JsonPropertyName("symbols")]
    public bool Symbols { get; init; }
}

public sealed class AnalysisResponse
{
    [JsonPropertyName("version")]
    public int Version { get; init; }

    [JsonPropertyName("language")]
    public string Language { get; init; } = string.Empty;

    [JsonPropertyName("breakpoints")]
    public IReadOnlyList<BreakpointDto> Breakpoints { get; init; } = [];

    [JsonPropertyName("symbols")]
    public IReadOnlyList<SymbolDto> Symbols { get; init; } = [];

    [JsonPropertyName("diagnostics")]
    public IReadOnlyList<string> Diagnostics { get; init; } = [];
}

public sealed class BreakpointDto
{
    [JsonPropertyName("pos")]
    public required int Pos { get; init; }

    [JsonPropertyName("type")]
    public required string Type { get; init; }

    [JsonPropertyName("score")]
    public required int Score { get; init; }
}

public sealed class SymbolDto
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("kind")]
    public required string Kind { get; init; }

    [JsonPropertyName("line")]
    public required int Line { get; init; }

    [JsonPropertyName("containerName")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ContainerName { get; init; }

    [JsonPropertyName("signature")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Signature { get; init; }

    [JsonPropertyName("modifiers")]
    public IReadOnlyList<string> Modifiers { get; init; } = [];
}
