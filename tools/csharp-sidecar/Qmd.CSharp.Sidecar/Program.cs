using System.Text.Json;

namespace Qmd.CSharp.Sidecar;

public static class Program
{
    public static async Task<int> Main()
    {
        await using var input = Console.OpenStandardInput();
        await using var output = Console.OpenStandardOutput();
        return await RunAsync(input, output, Console.Error);
    }

    public static async Task<int> RunAsync(Stream input, Stream output, TextWriter error)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(error);

        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web);
        AnalysisRequest? request;

        try
        {
            request = await JsonSerializer.DeserializeAsync<AnalysisRequest>(input, options);
        }
        catch (JsonException)
        {
            await error.WriteLineAsync("Invalid request payload.");
            return 1;
        }

        if (request is null)
        {
            await error.WriteLineAsync("Invalid request payload.");
            return 1;
        }

        var service = new CSharpAnalysisService();
        var response = service.Analyze(request);

        await JsonSerializer.SerializeAsync(output, response, options);
        return 0;
    }
}
