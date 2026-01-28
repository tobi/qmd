/**
 * Tests for the graph layer (entities, edges, traversal)
 */

import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { Database } from "bun:sqlite";
import {
  initGraphSchema,
  createEntity,
  getEntity,
  listEntities,
  searchEntities,
  deleteEntity,
  createEdge,
  getEdges,
  deleteEdge,
  traverse,
  findPath,
  getGraphStats,
  entityId,
} from "./graph";

describe("Graph Layer", () => {
  let db: Database;

  beforeEach(() => {
    db = new Database(":memory:");
    initGraphSchema(db);
  });

  afterEach(() => {
    db.close();
  });

  describe("entityId", () => {
    it("creates slugified entity IDs", () => {
      expect(entityId("person", "John Smith")).toBe("person:john-smith");
      expect(entityId("company", "Acme Corp")).toBe("company:acme-corp");
      expect(entityId("topic", "AI & ML")).toBe("topic:ai-ml");
    });
  });

  describe("Entity CRUD", () => {
    it("creates and retrieves entities", () => {
      const entity = createEntity(db, "person", "John Smith");
      expect(entity.id).toBe("person:john-smith");
      expect(entity.type).toBe("person");
      expect(entity.name).toBe("John Smith");

      const retrieved = getEntity(db, "person:john-smith");
      expect(retrieved).not.toBeNull();
      expect(retrieved!.name).toBe("John Smith");
    });

    it("creates entities with custom IDs", () => {
      const entity = createEntity(db, "person", "John", "custom-id");
      expect(entity.id).toBe("custom-id");
    });

    it("creates entities with metadata", () => {
      const entity = createEntity(db, "person", "John", undefined, { role: "founder" });
      expect(entity.metadata).toEqual({ role: "founder" });

      const retrieved = getEntity(db, "person:john");
      expect(retrieved!.metadata).toEqual({ role: "founder" });
    });

    it("upserts on conflict", () => {
      createEntity(db, "person", "John", undefined, { v: 1 });
      createEntity(db, "person", "John Updated", "person:john", { v: 2 });

      const entity = getEntity(db, "person:john");
      expect(entity!.name).toBe("John Updated");
      expect(entity!.metadata).toEqual({ v: 2 });
    });

    it("lists entities by type", () => {
      createEntity(db, "person", "John");
      createEntity(db, "person", "Joe");
      createEntity(db, "company", "Acme Corp");

      const people = listEntities(db, "person");
      expect(people.length).toBe(2);

      const companies = listEntities(db, "company");
      expect(companies.length).toBe(1);

      const all = listEntities(db);
      expect(all.length).toBe(3);
    });

    it("searches entities by name", () => {
      createEntity(db, "person", "John Smith");
      createEntity(db, "person", "Joe Bloggs");
      createEntity(db, "company", "Smith Industries");

      const results = searchEntities(db, "Smith");
      expect(results.length).toBe(2);
      expect(results.map(r => r.id)).toContain("person:john-smith");
    });

    it("deletes entities and their edges", () => {
      createEntity(db, "person", "John");
      createEntity(db, "company", "Acme Corp");
      createEdge(db, "person:john", "company:acme-corp", "owns");

      expect(getEdges(db, "person:john").length).toBe(1);

      deleteEntity(db, "person:john");

      expect(getEntity(db, "person:john")).toBeNull();
      expect(getEdges(db, "company:acme-corp").length).toBe(0);
    });
  });

  describe("Edge CRUD", () => {
    beforeEach(() => {
      createEntity(db, "person", "John");
      createEntity(db, "person", "Joe");
      createEntity(db, "company", "Acme Corp");
    });

    it("creates edges between entities", () => {
      const edge = createEdge(db, "person:john", "company:acme-corp", "owns");
      expect(edge.source_id).toBe("person:john");
      expect(edge.target_id).toBe("company:acme-corp");
      expect(edge.relation).toBe("owns");
      expect(edge.weight).toBe(1.0);
    });

    it("creates edges with weights", () => {
      const edge = createEdge(db, "person:john", "person:joe", "knows", 0.8);
      expect(edge.weight).toBe(0.8);
    });

    it("gets outgoing edges", () => {
      createEdge(db, "person:john", "company:acme-corp", "owns");
      createEdge(db, "person:joe", "company:acme-corp", "works_at");

      const outgoing = getEdges(db, "person:john", "outgoing");
      expect(outgoing.length).toBe(1);
      expect(outgoing[0]!.relation).toBe("owns");
    });

    it("gets incoming edges", () => {
      createEdge(db, "person:john", "company:acme-corp", "owns");
      createEdge(db, "person:joe", "company:acme-corp", "works_at");

      const incoming = getEdges(db, "company:acme-corp", "incoming");
      expect(incoming.length).toBe(2);
    });

    it("gets edges by relation", () => {
      createEdge(db, "person:john", "company:acme-corp", "owns");
      createEdge(db, "person:john", "person:joe", "knows");

      const owns = getEdges(db, "person:john", "both", "owns");
      expect(owns.length).toBe(1);
    });

    it("deletes specific edges", () => {
      createEdge(db, "person:john", "company:acme-corp", "owns");
      createEdge(db, "person:john", "company:acme-corp", "founded");

      deleteEdge(db, "person:john", "company:acme-corp", "owns");

      const remaining = getEdges(db, "person:john");
      expect(remaining.length).toBe(1);
      expect(remaining[0]!.relation).toBe("founded");
    });

    it("upserts edges on conflict", () => {
      createEdge(db, "person:john", "company:acme-corp", "owns", 0.5);
      createEdge(db, "person:john", "company:acme-corp", "owns", 1.0);

      const edges = getEdges(db, "person:john", "outgoing", "owns");
      expect(edges.length).toBe(1);
      expect(edges[0]!.weight).toBe(1.0);
    });
  });

  describe("Graph Traversal", () => {
    beforeEach(() => {
      // Create a small graph:
      // John --owns--> Acme Corp <--works_at-- Joe
      //   |                                      |
      //   +--knows--> Joe <--mentor-- Bob
      createEntity(db, "person", "John");
      createEntity(db, "person", "Joe");
      createEntity(db, "person", "Bob");
      createEntity(db, "company", "Acme Corp");

      createEdge(db, "person:john", "company:acme-corp", "owns");
      createEdge(db, "person:joe", "company:acme-corp", "works_at");
      createEdge(db, "person:john", "person:joe", "knows");
      createEdge(db, "person:bob", "person:joe", "mentors");
    });

    it("traverses from a node", () => {
      const results = traverse(db, "person:john", 1);
      expect(results.length).toBe(2); // Acme Corp and Joe
      expect(results.map(r => r.node_id)).toContain("company:acme-corp");
      expect(results.map(r => r.node_id)).toContain("person:joe");
    });

    it("traverses multiple hops", () => {
      const results = traverse(db, "person:john", 2);
      // Should find: Acme Corp (1 hop), Joe (1 hop), Bob (2 hops via Joe)
      expect(results.map(r => r.node_id)).toContain("person:bob");
    });

    it("respects direction constraints", () => {
      const outgoing = traverse(db, "person:john", 2, undefined, "outgoing");
      // Only follows outgoing edges from john: owns -> Acme Corp, knows -> Joe
      expect(outgoing.map(r => r.node_id)).toContain("company:acme-corp");
      expect(outgoing.map(r => r.node_id)).toContain("person:joe");
      // Bob's edge points TO Joe, so shouldn't be found in outgoing traversal from John
    });

    it("filters by relation types", () => {
      const results = traverse(db, "person:john", 2, ["owns"]);
      expect(results.length).toBe(1);
      expect(results[0]!.node_id).toBe("company:acme-corp");
    });

    it("tracks path and relations", () => {
      const results = traverse(db, "person:john", 2);
      const joe = results.find(r => r.node_id === "person:joe" && r.depth === 1);
      expect(joe).toBeDefined();
      expect(joe!.path).toEqual(["person:john", "person:joe"]);
      expect(joe!.relations).toEqual(["knows"]);
    });
  });

  describe("Path Finding", () => {
    beforeEach(() => {
      createEntity(db, "person", "A");
      createEntity(db, "person", "B");
      createEntity(db, "person", "C");
      createEntity(db, "person", "D");

      // A -> B -> C -> D
      createEdge(db, "person:a", "person:b", "next");
      createEdge(db, "person:b", "person:c", "next");
      createEdge(db, "person:c", "person:d", "next");
    });

    it("finds path between nodes", () => {
      const path = findPath(db, "person:a", "person:d");
      expect(path).not.toBeNull();
      expect(path!.depth).toBe(3);
      expect(path!.path).toEqual(["person:a", "person:b", "person:c", "person:d"]);
    });

    it("returns null when no path exists", () => {
      createEntity(db, "person", "Isolated");
      const path = findPath(db, "person:a", "person:isolated");
      expect(path).toBeNull();
    });

    it("respects max depth", () => {
      const path = findPath(db, "person:a", "person:d", 2);
      expect(path).toBeNull(); // Path requires 3 hops
    });
  });

  describe("Graph Statistics", () => {
    it("returns correct stats", () => {
      createEntity(db, "person", "John");
      createEntity(db, "person", "Joe");
      createEntity(db, "company", "Acme Corp");
      createEdge(db, "person:john", "company:acme-corp", "owns");
      createEdge(db, "person:joe", "company:acme-corp", "works_at");

      const stats = getGraphStats(db);
      expect(stats.entity_count).toBe(3);
      expect(stats.edge_count).toBe(2);
      expect(stats.entity_types).toEqual({ person: 2, company: 1 });
      expect(stats.relation_types).toEqual({ owns: 1, works_at: 1 });
    });

    it("handles empty graph", () => {
      const stats = getGraphStats(db);
      expect(stats.entity_count).toBe(0);
      expect(stats.edge_count).toBe(0);
      expect(stats.entity_types).toEqual({});
      expect(stats.relation_types).toEqual({});
    });
  });
});
