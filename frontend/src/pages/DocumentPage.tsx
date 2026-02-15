import { useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import * as api from "@/lib/api";
import { ArrowLeft, FileText, Clock, Hash } from "lucide-react";
import { Link } from "react-router-dom";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

export function DocumentPage() {
  const location = useLocation();
  const docPath = decodeURIComponent(location.pathname.replace(/^\/doc\//, ""));

  const { data: doc, isLoading, error } = useQuery({
    queryKey: ["doc", docPath],
    queryFn: () => api.getDocument(docPath),
    enabled: !!docPath,
  });

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto p-6 space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
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

  if (!doc) return null;

  return (
    <div className="max-w-4xl mx-auto p-6">
      <Link to="/">
        <Button variant="ghost" size="sm" className="mb-4">
          <ArrowLeft size={14} />
          Back to search
        </Button>
      </Link>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>{doc.title}</CardTitle>
          <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <FileText size={12} />
              {doc.displayPath}
            </span>
            <Badge variant="outline" className="font-mono text-[10px]">
              <Hash size={10} />
              {doc.docid}
            </Badge>
            <span className="flex items-center gap-1">
              <Clock size={12} />
              {new Date(doc.modifiedAt).toLocaleDateString()}
            </span>
            <Badge variant="secondary">{doc.collectionName}</Badge>
          </div>
          {doc.context && (
            <p className="text-xs text-muted-foreground italic mt-1">
              {doc.context}
            </p>
          )}
        </CardHeader>
      </Card>

      <article className="prose prose-invert prose-sm max-w-none">
        <ReactMarkdown rehypePlugins={[rehypeHighlight]} remarkPlugins={[remarkGfm]}>
          {doc.body}
        </ReactMarkdown>
      </article>
    </div>
  );
}
