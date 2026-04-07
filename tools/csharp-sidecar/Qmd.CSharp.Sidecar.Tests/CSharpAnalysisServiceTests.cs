using System.Text;
using System.Text.Json;
using Qmd.CSharp.Sidecar;
using Xunit;

namespace Qmd.CSharp.Sidecar.Tests;

public sealed class CSharpAnalysisServiceTests
{
    [Fact]
    public void Analyze_returns_protocol_envelope_for_simple_csharp()
    {
        const string content = """
            using System;

            namespace Demo.App;

            public class Greeter
            {
                public Greeter()
                {
                }

                public void SayHello()
                {
                    Console.WriteLine("hi");
                }
            }
            """;

        var service = new CSharpAnalysisService();
        var request = new AnalysisRequest
        {
            FilePath = "/workspace/Greeter.cs",
            Content = content,
            Features = new AnalysisFeatures
            {
                Breakpoints = true
            }
        };

        var response = service.Analyze(request);
        var types = response.Breakpoints.Select(static breakpoint => breakpoint.Type).ToArray();

        Assert.Equal(1, response.Version);
        Assert.Equal("csharp", response.Language);
        Assert.Equal(
            ["roslyn:import", "roslyn:namespace", "roslyn:type", "roslyn:ctor", "roslyn:method"],
            types);
        Assert.Empty(response.Symbols);
        Assert.Empty(response.Diagnostics);
    }

    [Fact]
    public void Analyze_serializes_response_with_expected_protocol_property_names()
    {
        const string content = "using System;";

        var service = new CSharpAnalysisService();
        var request = new AnalysisRequest
        {
            Version = 1,
            Language = "csharp",
            FilePath = "/workspace/Greeter.cs",
            Content = content,
            Features = new AnalysisFeatures
            {
                Breakpoints = true,
                Symbols = true
            }
        };

        var response = service.Analyze(request);
        var json = JsonSerializer.Serialize(response, new JsonSerializerOptions(JsonSerializerDefaults.Web));
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;

        Assert.Equal(1, root.GetProperty("version").GetInt32());
        Assert.Equal("csharp", root.GetProperty("language").GetString());
        Assert.True(root.TryGetProperty("breakpoints", out var breakpointsElement));
        Assert.True(root.TryGetProperty("symbols", out var symbolsElement));
        Assert.True(root.TryGetProperty("diagnostics", out var diagnosticsElement));
        Assert.Equal(JsonValueKind.Array, breakpointsElement.ValueKind);
        Assert.Equal(JsonValueKind.Array, symbolsElement.ValueKind);
        Assert.Equal(JsonValueKind.Array, diagnosticsElement.ValueKind);

        var breakpoint = Assert.Single(breakpointsElement.EnumerateArray());
        Assert.Equal(0, breakpoint.GetProperty("pos").GetInt32());
        Assert.Equal("roslyn:import", breakpoint.GetProperty("type").GetString());
        Assert.Equal(60, breakpoint.GetProperty("score").GetInt32());
        Assert.False(breakpoint.TryGetProperty("kind", out _));
    }

    [Fact]
    public void AnalysisRequest_deserializes_expected_protocol_fields()
    {
        const string json = """
            {
              "version": 1,
              "language": "csharp",
              "filePath": "/workspace/Greeter.cs",
              "content": "using System;",
              "features": {
                "breakpoints": true,
                "symbols": false
              }
            }
            """;

        var request = JsonSerializer.Deserialize<AnalysisRequest>(
            json,
            new JsonSerializerOptions(JsonSerializerDefaults.Web));

        Assert.NotNull(request);
        Assert.Equal(1, request.Version);
        Assert.Equal("csharp", request.Language);
        Assert.Equal("/workspace/Greeter.cs", request.FilePath);
        Assert.Equal("using System;", request.Content);
        Assert.True(request.Features?.Breakpoints);
        Assert.False(request.Features?.Symbols);
    }

    [Fact]
    public void SymbolDto_serializes_expected_protocol_fields()
    {
        var symbol = new SymbolDto
        {
            Name = "SayHello",
            Kind = "method",
            Line = 10,
            ContainerName = "Greeter",
            Signature = "void SayHello()",
            Modifiers = ["public"]
        };

        var json = JsonSerializer.Serialize(symbol, new JsonSerializerOptions(JsonSerializerDefaults.Web));
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;

        Assert.Equal("SayHello", root.GetProperty("name").GetString());
        Assert.Equal("method", root.GetProperty("kind").GetString());
        Assert.Equal(10, root.GetProperty("line").GetInt32());
        Assert.Equal("Greeter", root.GetProperty("containerName").GetString());
        Assert.Equal("void SayHello()", root.GetProperty("signature").GetString());
        Assert.Equal("public", Assert.Single(root.GetProperty("modifiers").EnumerateArray()).GetString());
    }

    [Fact]
    public async Task Program_returns_error_for_invalid_request_payload()
    {
        await using var input = new MemoryStream(Encoding.UTF8.GetBytes(string.Empty));
        await using var output = new MemoryStream();
        await using var error = new StringWriter();

        var exitCode = await Program.RunAsync(input, output, error);

        Assert.Equal(1, exitCode);
        Assert.Equal("Invalid request payload." + Environment.NewLine, error.ToString());
        Assert.Equal(0, output.Length);
    }
}
