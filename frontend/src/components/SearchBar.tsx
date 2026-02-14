import { useState, type FormEvent } from "react";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

export type SearchTier = "keyword" | "semantic" | "deep";

interface SearchBarProps {
  onSearch: (query: string, tier: SearchTier) => void;
  loading?: boolean;
  initialQuery?: string;
  initialTier?: SearchTier;
}

export function SearchBar({
  onSearch,
  loading,
  initialQuery = "",
  initialTier = "keyword",
}: SearchBarProps) {
  const [query, setQuery] = useState(initialQuery);
  const [tier, setTier] = useState<SearchTier>(initialTier);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) onSearch(query.trim(), tier);
  };

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="relative flex-1">
          <Search
            className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground"
            size={16}
          />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search documents..."
            className="pl-10"
          />
        </div>
        <Button type="submit" disabled={loading || !query.trim()}>
          {loading ? "Searching..." : "Search"}
        </Button>
      </form>
      <Tabs
        value={tier}
        onValueChange={(v) => setTier(v as SearchTier)}
      >
        <TabsList>
          <TabsTrigger value="keyword">Keyword</TabsTrigger>
          <TabsTrigger value="semantic">Semantic</TabsTrigger>
          <TabsTrigger value="deep">Deep</TabsTrigger>
        </TabsList>
      </Tabs>
    </div>
  );
}
