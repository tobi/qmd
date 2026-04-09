using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System.Text.RegularExpressions;

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

        var symbols = new List<SymbolDto>();

        foreach (var node in root.DescendantNodesAndSelf())
        {
            switch (node)
            {
                case ClassDeclarationSyntax classDeclaration:
                    symbols.Add(CreateSymbol(classDeclaration.Identifier.ValueText, "class", classDeclaration));
                    break;
                case StructDeclarationSyntax structDeclaration:
                    symbols.Add(CreateSymbol(structDeclaration.Identifier.ValueText, "struct", structDeclaration));
                    break;
                case InterfaceDeclarationSyntax interfaceDeclaration:
                    symbols.Add(CreateSymbol(interfaceDeclaration.Identifier.ValueText, "interface", interfaceDeclaration));
                    break;
                case RecordDeclarationSyntax recordDeclaration:
                    symbols.Add(CreateSymbol(recordDeclaration.Identifier.ValueText, "record", recordDeclaration));
                    break;
                case EnumDeclarationSyntax enumDeclaration:
                    symbols.Add(CreateSymbol(enumDeclaration.Identifier.ValueText, "enum", enumDeclaration));
                    break;
                case ConstructorDeclarationSyntax constructorDeclaration:
                    symbols.Add(CreateSymbol(constructorDeclaration.Identifier.ValueText, "constructor", constructorDeclaration));
                    break;
                case MethodDeclarationSyntax methodDeclaration:
                    symbols.Add(CreateSymbol(methodDeclaration.Identifier.ValueText, "method", methodDeclaration));
                    break;
                case PropertyDeclarationSyntax propertyDeclaration:
                    symbols.Add(CreateSymbol(propertyDeclaration.Identifier.ValueText, "property", propertyDeclaration));
                    break;
            }
        }

        return symbols;
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

    private static SymbolDto CreateSymbol(string name, string kind, MemberDeclarationSyntax declaration)
    {
        var line = declaration.SyntaxTree.GetLineSpan(declaration.Span).StartLinePosition.Line + 1;
        var containerName = GetContainerName(declaration.Parent);
        var signature = GetSignature(declaration);

        return new SymbolDto
        {
            Name = name,
            Kind = kind,
            Line = line,
            ContainerName = containerName,
            Signature = string.IsNullOrWhiteSpace(signature) ? null : signature,
            Modifiers = declaration.GetModifierTexts()
        };
    }

    private static string? GetContainerName(SyntaxNode? parent)
    {
        while (parent is not null)
        {
            switch (parent)
            {
                case BaseNamespaceDeclarationSyntax namespaceDeclaration:
                    return namespaceDeclaration.Name.ToString();
                case BaseTypeDeclarationSyntax typeDeclaration:
                    return typeDeclaration.Identifier.ValueText;
            }

            parent = parent.Parent;
        }

        return null;
    }

    private static string? GetSignature(MemberDeclarationSyntax declaration)
    {
        var headerLines = declaration
            .ToString()
            .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .SkipWhile(static line => line.StartsWith("[", StringComparison.Ordinal));

        var header = Regex.Replace(string.Join(" ", headerLines), "\\s+", " ").Trim();
        if (string.IsNullOrWhiteSpace(header))
        {
            return null;
        }

        var cutIndex = FindHeaderTerminator(header);
        return cutIndex >= 0
            ? header[..cutIndex].TrimEnd()
            : header;
    }

    private static int FindHeaderTerminator(string header)
    {
        var arrowIndex = header.IndexOf("=>", StringComparison.Ordinal);
        if (arrowIndex >= 0)
        {
            return arrowIndex;
        }

        var blockIndex = header.IndexOf('{');
        if (blockIndex >= 0)
        {
            return blockIndex;
        }

        var semicolonIndex = header.IndexOf(';');
        return semicolonIndex;
    }
}

internal static class MemberDeclarationSyntaxExtensions
{
    public static IReadOnlyList<string> GetModifierTexts(this MemberDeclarationSyntax declaration) =>
        GetModifiers(declaration).Select(static modifier => modifier.ValueText).ToArray();

    private static SyntaxTokenList GetModifiers(MemberDeclarationSyntax declaration) =>
        declaration switch
        {
            BaseTypeDeclarationSyntax typeDeclaration => typeDeclaration.Modifiers,
            BaseMethodDeclarationSyntax methodDeclaration => methodDeclaration.Modifiers,
            PropertyDeclarationSyntax propertyDeclaration => propertyDeclaration.Modifiers,
            EventDeclarationSyntax eventDeclaration => eventDeclaration.Modifiers,
            FieldDeclarationSyntax fieldDeclaration => fieldDeclaration.Modifiers,
            DelegateDeclarationSyntax delegateDeclaration => delegateDeclaration.Modifiers,
            EnumMemberDeclarationSyntax => default,
            _ => default
        };
}
