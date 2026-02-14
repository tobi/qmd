import { Link } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { SearchResult, HybridQueryResult } from "@/lib/api";

interface ResultCardProps {
  result: SearchResult | HybridQueryResult;
}

function isSearchResult(r: SearchResult | HybridQueryResult): r is SearchResult {
  return "source" in r;
}

export function ResultCard({ result }: ResultCardProps) {
  const displayPath = result.displayPath;
  const docid = result.docid;
  const score = result.score;
  const title = result.title;
  const snippet = isSearchResult(result) ? result.body : result.bestChunk;

  return (
    <Link to={`/doc/${encodeURIComponent(displayPath)}`}>
      <Card className="hover:border-primary/50 transition-colors">
        <CardContent className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <h3 className="font-medium text-sm truncate">{title}</h3>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-muted-foreground truncate">
                  {displayPath}
                </span>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0 font-mono">
                  #{docid}
                </Badge>
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {isSearchResult(result) && (
                <Badge variant="secondary" className="text-[10px] uppercase">
                  {result.source}
                </Badge>
              )}
              <Badge className="font-mono">
                {(score * 100).toFixed(0)}%
              </Badge>
            </div>
          </div>
          {snippet && (
            <p className="mt-2 text-xs text-muted-foreground line-clamp-3 whitespace-pre-wrap">
              {snippet.slice(0, 300)}
            </p>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}
