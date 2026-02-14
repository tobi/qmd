import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { SearchBar, type SearchTier } from "@/components/SearchBar";
import { ResultCard } from "@/components/ResultCard";
import * as api from "@/lib/api";
import type { SearchResult, HybridQueryResult } from "@/lib/api";

type AnyResult = SearchResult | HybridQueryResult;

export function SearchPage() {
  const [searchParams, setSearchParams] = useState<{
    query: string;
    tier: SearchTier;
  } | null>(null);

  const { data, isLoading, error } = useQuery<AnyResult[] | null>({
    queryKey: ["search", searchParams?.query, searchParams?.tier],
    queryFn: async (): Promise<AnyResult[] | null> => {
      if (!searchParams) return null;
      const { query, tier } = searchParams;
      switch (tier) {
        case "keyword":
          return api.search(query);
        case "semantic":
          return api.vsearch(query);
        case "deep":
          return api.deepSearch(query);
      }
    },
    enabled: !!searchParams,
  });

  const handleSearch = (query: string, tier: SearchTier) => {
    setSearchParams({ query, tier });
  };

  const results: AnyResult[] = data ?? [];

  return (
    <div className="max-w-3xl mx-auto p-6">
      <SearchBar
        onSearch={handleSearch}
        loading={isLoading}
        initialQuery={searchParams?.query}
        initialTier={searchParams?.tier}
      />

      {error && (
        <div className="mt-4 rounded-md bg-destructive/10 border border-destructive/20 p-3 text-sm text-destructive">
          {(error as Error).message}
        </div>
      )}

      {isLoading && (
        <div className="mt-8 text-center text-muted-foreground text-sm">
          {searchParams?.tier === "deep"
            ? "Running hybrid search with reranking..."
            : searchParams?.tier === "semantic"
            ? "Running vector search..."
            : "Searching..."}
        </div>
      )}

      {!isLoading && results.length > 0 && (
        <div className="mt-6 space-y-3">
          <div className="text-xs text-muted-foreground">
            {results.length} result{results.length !== 1 ? "s" : ""}
          </div>
          {results.map((r: AnyResult, i: number) => (
            <ResultCard key={`${r.docid}-${i}`} result={r} />
          ))}
        </div>
      )}

      {!isLoading && searchParams && results.length === 0 && (
        <div className="mt-8 text-center text-muted-foreground text-sm">
          No results found.
        </div>
      )}

      {!searchParams && (
        <div className="mt-16 text-center text-muted-foreground">
          <p className="text-lg mb-2">Search your documents</p>
          <p className="text-sm">
            Use <strong>Keyword</strong> for fast BM25 search,{" "}
            <strong>Semantic</strong> for meaning-based search, or{" "}
            <strong>Deep</strong> for hybrid search with LLM reranking.
          </p>
        </div>
      )}
    </div>
  );
}
