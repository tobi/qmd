import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams, Link } from "react-router-dom";
import {
  FolderOpen,
  FileText,
  ChevronRight,
  Trash2,
  MessageSquare,
  Plus,
  X,
} from "lucide-react";
import * as api from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";

export function BrowsePage() {
  const params = useParams();
  const collection = params.collection;
  const subPath = params["*"];
  const lsPath = collection
    ? subPath
      ? `${collection}/${subPath}`
      : collection
    : undefined;

  const queryClient = useQueryClient();

  const { data: lsResult, isLoading } = useQuery({
    queryKey: ["ls", lsPath],
    queryFn: () => api.listFiles(lsPath),
  });

  const { data: contexts } = useQuery({
    queryKey: ["contexts"],
    queryFn: api.getContexts,
  });

  const [addingContext, setAddingContext] = useState<string | null>(null);
  const [contextText, setContextText] = useState("");

  const addContextMut = useMutation({
    mutationFn: ({
      coll,
      path,
      text,
    }: {
      coll: string;
      path: string;
      text: string;
    }) => api.addContext(coll, path, text),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["contexts"] });
      setAddingContext(null);
      setContextText("");
    },
  });

  const removeContextMut = useMutation({
    mutationFn: ({ coll, path }: { coll: string; path: string }) =>
      api.removeContext(coll, path),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["contexts"] }),
  });

  const removeCollectionMut = useMutation({
    mutationFn: (name: string) => api.removeCollection(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["ls"] });
      queryClient.invalidateQueries({ queryKey: ["collections"] });
    },
  });

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto p-6 space-y-3">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-16 w-full" />
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1 text-sm text-muted-foreground mb-4">
        <Link to="/browse" className="hover:text-foreground">
          Collections
        </Link>
        {collection && (
          <>
            <ChevronRight size={14} />
            <Link
              to={`/browse/${collection}`}
              className="hover:text-foreground"
            >
              {collection}
            </Link>
          </>
        )}
        {subPath && (
          <>
            <ChevronRight size={14} />
            <span>{subPath}</span>
          </>
        )}
      </div>

      {lsResult?.type === "collections" && (
        <div className="space-y-2">
          <h2 className="text-lg font-semibold mb-4">Collections</h2>
          {lsResult.collections.map((c) => {
            const collContexts = contexts?.filter(
              (ctx) => ctx.collection === c.name
            );
            return (
              <Card key={c.name}>
                <CardContent className="p-0">
                  <div className="flex items-center justify-between p-3">
                    <Link
                      to={`/browse/${c.name}`}
                      className="flex items-center gap-2 hover:text-primary"
                    >
                      <FolderOpen size={16} className="text-primary" />
                      <span className="font-medium text-sm">{c.name}</span>
                      <Badge variant="secondary" className="text-[10px]">
                        {c.file_count} files
                      </Badge>
                    </Link>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() =>
                          setAddingContext(
                            addingContext === c.name ? null : c.name
                          )
                        }
                        title="Add context"
                      >
                        <MessageSquare size={14} />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 hover:text-destructive"
                        onClick={() => {
                          if (confirm(`Remove collection "${c.name}"?`))
                            removeCollectionMut.mutate(c.name);
                        }}
                        title="Remove collection"
                      >
                        <Trash2 size={14} />
                      </Button>
                    </div>
                  </div>

                  {collContexts && collContexts.length > 0 && (
                    <>
                      <Separator />
                      <div className="px-3 py-2 space-y-1">
                        {collContexts.map((ctx) => (
                          <div
                            key={ctx.path}
                            className="flex items-center justify-between text-xs"
                          >
                            <span className="text-muted-foreground">
                              <Badge variant="outline" className="text-[10px] font-mono mr-1 px-1 py-0">
                                {ctx.path}
                              </Badge>
                              {ctx.context}
                            </span>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-5 w-5 hover:text-destructive"
                              onClick={() =>
                                removeContextMut.mutate({
                                  coll: c.name,
                                  path: ctx.path,
                                })
                              }
                            >
                              <X size={12} />
                            </Button>
                          </div>
                        ))}
                      </div>
                    </>
                  )}

                  {addingContext === c.name && (
                    <>
                      <Separator />
                      <div className="px-3 py-2">
                        <form
                          onSubmit={(e) => {
                            e.preventDefault();
                            if (contextText.trim()) {
                              addContextMut.mutate({
                                coll: c.name,
                                path: "/",
                                text: contextText.trim(),
                              });
                            }
                          }}
                          className="flex gap-2"
                        >
                          <Input
                            value={contextText}
                            onChange={(e) => setContextText(e.target.value)}
                            placeholder="Context description..."
                            className="h-8 text-xs"
                          />
                          <Button type="submit" size="sm" className="h-8">
                            <Plus size={12} />
                          </Button>
                        </form>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {lsResult?.type === "files" && (
        <div className="space-y-1">
          <h2 className="text-lg font-semibold mb-4">
            {lsResult.collection}
            {lsResult.path ? ` / ${lsResult.path}` : ""}
          </h2>
          {lsResult.files.length === 0 && (
            <p className="text-sm text-muted-foreground">No files found.</p>
          )}
          {lsResult.files.map((f) => (
            <Link
              key={f.displayPath}
              to={`/doc/${encodeURIComponent(f.displayPath)}`}
              className="flex items-center justify-between rounded-md px-3 py-2 hover:bg-muted transition-colors"
            >
              <div className="flex items-center gap-2 min-w-0">
                <FileText size={14} className="text-muted-foreground shrink-0" />
                <div className="min-w-0">
                  <div className="text-sm truncate">{f.title}</div>
                  <div className="text-xs text-muted-foreground truncate">
                    {f.displayPath}
                  </div>
                </div>
              </div>
              <Badge variant="outline" className="text-[10px] font-mono shrink-0 ml-2">
                #{f.docid}
              </Badge>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
