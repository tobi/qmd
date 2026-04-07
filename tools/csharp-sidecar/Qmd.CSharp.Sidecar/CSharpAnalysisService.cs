using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Qmd.CSharp.Sidecar;

public sealed class CSharpAnalysisService
{
    public AnalysisResponse Analyze(AnalysisRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        var tree = CSharpSyntaxTree.ParseText(request.Content, path: request.FilePath);
        var root = tree.GetCompilationUnitRoot();

        return new AnalysisResponse
        {
            Version = 1,
            Language = "csharp",
            Breakpoints = request.Features?.Breakpoints == true ? CollectBreakpoints(root) : [],
            Symbols = request.Features?.Symbols == true ? CollectSymbols(root) : [],
            Diagnostics = []
        };
    }

    public IReadOnlyList<BreakpointDto> CollectBreakpoints(SyntaxNode root)
    {
        ArgumentNullException.ThrowIfNull(root);

        var breakpoints = new Dictionary<int, BreakpointDto>();

        foreach (var node in root.DescendantNodesAndSelf())
        {
            switch (node)
            {
                case UsingDirectiveSyntax usingDirective:
                    AddOrUpdateBreakpoint(breakpoints, usingDirective.SpanStart, "roslyn:import", 60);
                    break;
                case BaseNamespaceDeclarationSyntax namespaceDeclaration:
                    AddOrUpdateBreakpoint(breakpoints, namespaceDeclaration.SpanStart, "roslyn:namespace", 100);
                    break;
                case EnumDeclarationSyntax enumDeclaration:
                    AddOrUpdateBreakpoint(breakpoints, enumDeclaration.SpanStart, "roslyn:enum", 80);
                    break;
                case RecordDeclarationSyntax recordDeclaration:
                    AddOrUpdateBreakpoint(breakpoints, recordDeclaration.SpanStart, "roslyn:type", 100);
                    break;
                case TypeDeclarationSyntax typeDeclaration:
                    AddOrUpdateBreakpoint(breakpoints, typeDeclaration.SpanStart, "roslyn:type", 100);
                    break;
                case ConstructorDeclarationSyntax constructorDeclaration:
                    AddOrUpdateBreakpoint(breakpoints, constructorDeclaration.SpanStart, "roslyn:ctor", 90);
                    break;
                case MethodDeclarationSyntax methodDeclaration:
                    AddOrUpdateBreakpoint(breakpoints, methodDeclaration.SpanStart, "roslyn:method", 90);
                    break;
            }
        }

        return breakpoints.Values.OrderBy(static breakpoint => breakpoint.Pos).ToArray();
    }

    public IReadOnlyList<SymbolDto> CollectSymbols(SyntaxNode root)
    {
        ArgumentNullException.ThrowIfNull(root);
        return [];
    }

    private static void AddOrUpdateBreakpoint(
        IDictionary<int, BreakpointDto> breakpoints,
        int pos,
        string type,
        int score)
    {
        if (!breakpoints.TryGetValue(pos, out var existing) || score > existing.Score)
        {
            breakpoints[pos] = new BreakpointDto
            {
                Pos = pos,
                Type = type,
                Score = score
            };
        }
    }
}
