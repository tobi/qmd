using Qmd.CSharp.Sidecar;
using Xunit;

namespace Qmd.CSharp.Sidecar.Tests;

public sealed class CSharpAnalysisServiceTests
{
    [Fact]
    public void Analyze_collects_expected_breakpoint_kinds_for_simple_csharp()
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
        var kinds = response.Breakpoints.Select(static breakpoint => breakpoint.Kind).ToArray();

        Assert.Equal(
            ["roslyn:import", "roslyn:namespace", "roslyn:type", "roslyn:ctor", "roslyn:method"],
            kinds);
        Assert.Empty(response.Symbols);
    }
}
