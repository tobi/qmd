import { useQuery } from "@tanstack/react-query";
import {
  Database,
  FileText,
  Layers,
  AlertTriangle,
  CheckCircle,
} from "lucide-react";
import * as api from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

export function StatusPage() {
  const { data: status, isLoading, error } = useQuery({
    queryKey: ["status"],
    queryFn: api.getStatus,
    refetchInterval: 10_000,
  });

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto p-6 space-y-4">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
        </div>
        <Skeleton className="h-48" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <Card className="border-destructive/20 bg-destructive/10 p-4">
          <p className="text-sm text-destructive">{(error as Error).message}</p>
        </Card>
      </div>
    );
  }

  if (!status) return null;

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-xl font-semibold mb-6">Index Status</h1>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <FileText size={16} />
              <span className="text-xs uppercase tracking-wide">Documents</span>
            </div>
            <div className="text-2xl font-bold">{status.totalDocuments}</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Layers size={16} />
              <span className="text-xs uppercase tracking-wide">Collections</span>
            </div>
            <div className="text-2xl font-bold">{status.collections.length}</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Database size={16} />
              <span className="text-xs uppercase tracking-wide">Vector Index</span>
            </div>
            <div className="flex items-center gap-2">
              {status.hasVectorIndex ? (
                <>
                  <CheckCircle size={18} className="text-green-500" />
                  <span className="text-sm">Active</span>
                </>
              ) : (
                <>
                  <AlertTriangle size={18} className="text-yellow-500" />
                  <span className="text-sm">Not built</span>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Embedding status */}
      {status.needsEmbedding > 0 && (
        <Card className="border-yellow-500/20 bg-yellow-500/10 mb-6">
          <CardContent className="p-3 flex items-center gap-2 text-sm text-yellow-400">
            <AlertTriangle size={16} />
            {status.needsEmbedding} document{status.needsEmbedding !== 1 ? "s" : ""}{" "}
            need embedding. Run <code className="mx-1 font-mono">qmd embed</code> to generate.
          </CardContent>
        </Card>
      )}

      {/* Health info */}
      {status.health && status.health.daysStale !== null && status.health.daysStale > 7 && (
        <Card className="border-yellow-500/20 bg-yellow-500/10 mb-6">
          <CardContent className="p-3 flex items-center gap-2 text-sm text-yellow-400">
            <AlertTriangle size={16} />
            Index is {status.health.daysStale} days old. Run{" "}
            <code className="mx-1 font-mono">qmd update</code> to refresh.
          </CardContent>
        </Card>
      )}

      {/* Collection details */}
      <Card>
        <CardHeader>
          <CardTitle>Collections</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-muted/50">
                <th className="text-left px-4 py-2 font-medium">Name</th>
                <th className="text-left px-4 py-2 font-medium">Path</th>
                <th className="text-left px-4 py-2 font-medium">Pattern</th>
                <th className="text-right px-4 py-2 font-medium">Docs</th>
                <th className="text-right px-4 py-2 font-medium">Last Updated</th>
              </tr>
            </thead>
            <tbody>
              {status.collections.map((c) => (
                <tr key={c.name} className="border-t border-border">
                  <td className="px-4 py-2">
                    <Badge variant="secondary">{c.name}</Badge>
                  </td>
                  <td className="px-4 py-2 text-muted-foreground truncate max-w-48">
                    {c.path}
                  </td>
                  <td className="px-4 py-2">
                    <Badge variant="outline" className="font-mono text-[10px]">
                      {c.pattern}
                    </Badge>
                  </td>
                  <td className="px-4 py-2 text-right">{c.documents}</td>
                  <td className="px-4 py-2 text-right text-muted-foreground">
                    {new Date(c.lastUpdated).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}
