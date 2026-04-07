/**
 * ast.test.ts - Tests for AST-aware chunking support
 *
 * Tests language detection, AST break point extraction for each
 * supported language, and graceful fallback on errors.
 */

import { describe, test, expect } from "vitest";
import { detectLanguage, getASTBreakPoints, extractSymbols } from "../src/ast.js";
import type { SupportedLanguage } from "../src/ast.js";

// =============================================================================
// Language Detection
// =============================================================================

describe("detectLanguage", () => {
  test("recognizes TypeScript extensions", () => {
    expect(detectLanguage("src/auth.ts")).toBe("typescript");
    expect(detectLanguage("src/auth.mts")).toBe("typescript");
    expect(detectLanguage("src/auth.cts")).toBe("typescript");
  });

  test("recognizes TSX extension", () => {
    expect(detectLanguage("src/App.tsx")).toBe("tsx");
  });

  test("recognizes JavaScript extensions", () => {
    expect(detectLanguage("src/util.js")).toBe("javascript");
    expect(detectLanguage("src/util.mjs")).toBe("javascript");
    expect(detectLanguage("src/util.cjs")).toBe("javascript");
  });

  test("recognizes JSX as tsx", () => {
    expect(detectLanguage("src/App.jsx")).toBe("tsx");
  });

  test("recognizes Python extension", () => {
    expect(detectLanguage("src/auth.py")).toBe("python");
  });

  test("recognizes Go extension", () => {
    expect(detectLanguage("src/auth.go")).toBe("go");
  });

  test("recognizes Rust extension", () => {
    expect(detectLanguage("src/auth.rs")).toBe("rust");
  });

  test("recognizes C# extension", () => {
    expect(detectLanguage("src/AuthService.cs")).toBe("csharp");
  });

  test("returns null for markdown", () => {
    expect(detectLanguage("docs/README.md")).toBeNull();
  });

  test("returns null for unknown extensions", () => {
    expect(detectLanguage("data/file.csv")).toBeNull();
    expect(detectLanguage("config.yaml")).toBeNull();
    expect(detectLanguage("Makefile")).toBeNull();
  });

  test("is case-insensitive for extensions", () => {
    expect(detectLanguage("src/Auth.TS")).toBe("typescript");
    expect(detectLanguage("src/Auth.PY")).toBe("python");
  });

  test("works with virtual qmd:// paths", () => {
    expect(detectLanguage("qmd://myproject/src/auth.ts")).toBe("typescript");
    expect(detectLanguage("qmd://docs/README.md")).toBeNull();
  });
});

// =============================================================================
// AST Break Points - TypeScript
// =============================================================================

describe("getASTBreakPoints - TypeScript", () => {
  const TS_SAMPLE = `import { Database } from './db';
import type { User } from './types';

interface AuthConfig {
  secret: string;
  ttl: number;
}

type UserId = string;

export class AuthService {
  constructor(private db: Database) {}

  async authenticate(user: User, token: string): Promise<boolean> {
    const session = await this.db.findSession(token);
    return session?.userId === user.id;
  }

  validateToken(token: string): boolean {
    return token.length === 64;
  }
}

export function hashPassword(password: string): string {
  return crypto.createHash('sha256').update(password).digest('hex');
}
`;

  test("produces break points at function, class, and import boundaries", async () => {
    const points = await getASTBreakPoints(TS_SAMPLE, "src/auth.ts");
    expect(points.length).toBeGreaterThan(0);

    // Should have import, interface, type, class (via export), method, and function break points
    const types = points.map(p => p.type);
    expect(types.some(t => t.includes("import"))).toBe(true);
    expect(types.some(t => t.includes("iface"))).toBe(true);
    expect(types.some(t => t.includes("type"))).toBe(true);
    expect(types.some(t => t.includes("export") || t.includes("class"))).toBe(true);
    expect(types.some(t => t.includes("method"))).toBe(true);
  });

  test("break points are sorted by position", async () => {
    const points = await getASTBreakPoints(TS_SAMPLE, "src/auth.ts");
    for (let i = 1; i < points.length; i++) {
      expect(points[i]!.pos).toBeGreaterThanOrEqual(points[i - 1]!.pos);
    }
  });

  test("scores align with expected hierarchy", async () => {
    const points = await getASTBreakPoints(TS_SAMPLE, "src/auth.ts");

    // Class/interface should score 100
    const ifacePoint = points.find(p => p.type === "ast:iface");
    expect(ifacePoint?.score).toBe(100);

    // Function/method should score 90
    const methodPoint = points.find(p => p.type === "ast:method");
    expect(methodPoint?.score).toBe(90);

    // Import should score 60
    const importPoint = points.find(p => p.type === "ast:import");
    expect(importPoint?.score).toBe(60);
  });

  test("break point positions match actual content positions", async () => {
    const points = await getASTBreakPoints(TS_SAMPLE, "src/auth.ts");

    // First import should be at position 0
    const firstImport = points.find(p => p.type === "ast:import");
    expect(firstImport).toBeDefined();
    expect(TS_SAMPLE.slice(firstImport!.pos, firstImport!.pos + 6)).toBe("import");
  });
});

// =============================================================================
// AST Break Points - Python
// =============================================================================

describe("getASTBreakPoints - Python", () => {
  const PY_SAMPLE = `import os
from typing import Optional

class AuthService:
    def __init__(self, db):
        self.db = db

    async def authenticate(self, user, token):
        session = await self.db.find(token)
        return session.user_id == user.id

    def validate_token(self, token):
        return len(token) == 64

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

@decorator
def decorated_func():
    pass
`;

  test("produces break points for class, function, import, and decorated definitions", async () => {
    const points = await getASTBreakPoints(PY_SAMPLE, "auth.py");
    const types = points.map(p => p.type);

    expect(types.some(t => t.includes("import"))).toBe(true);
    expect(types.some(t => t.includes("class"))).toBe(true);
    expect(types.some(t => t.includes("func"))).toBe(true);
    expect(types.some(t => t.includes("decorated"))).toBe(true);
  });

  test("captures method definitions inside classes", async () => {
    const points = await getASTBreakPoints(PY_SAMPLE, "auth.py");
    // Should capture __init__, authenticate, and validate_token as func
    const funcPoints = points.filter(p => p.type === "ast:func");
    expect(funcPoints.length).toBeGreaterThanOrEqual(3);
  });
});

// =============================================================================
// AST Break Points - Go
// =============================================================================

describe("getASTBreakPoints - Go", () => {
  const GO_SAMPLE = `package main

import "fmt"

type AuthService struct {
    db *Database
}

func (s *AuthService) Authenticate(user User) bool {
    return true
}

func HashPassword(password string) string {
    return "hash"
}
`;

  test("produces break points for type, function, method, and import", async () => {
    const points = await getASTBreakPoints(GO_SAMPLE, "auth.go");
    const types = points.map(p => p.type);

    expect(types.some(t => t.includes("import"))).toBe(true);
    expect(types.some(t => t.includes("type"))).toBe(true);
    expect(types.some(t => t.includes("method"))).toBe(true);
    expect(types.some(t => t.includes("func"))).toBe(true);
  });

  test("function and method both score 90", async () => {
    const points = await getASTBreakPoints(GO_SAMPLE, "auth.go");
    const funcPoint = points.find(p => p.type === "ast:func");
    const methodPoint = points.find(p => p.type === "ast:method");

    expect(funcPoint?.score).toBe(90);
    expect(methodPoint?.score).toBe(90);
  });
});

// =============================================================================
// AST Break Points - Rust
// =============================================================================

describe("getASTBreakPoints - Rust", () => {
  const RS_SAMPLE = `use std::collections::HashMap;

struct AuthService {
    db: Database,
}

impl AuthService {
    fn authenticate(&self, user: &User) -> bool {
        true
    }
}

trait Authenticatable {
    fn validate(&self) -> bool;
}

enum Role {
    Admin,
    User,
}

fn hash_password(password: &str) -> String {
    String::new()
}
`;

  test("produces break points for struct, impl, trait, enum, function, and use", async () => {
    const points = await getASTBreakPoints(RS_SAMPLE, "auth.rs");
    const types = points.map(p => p.type);

    expect(types.some(t => t.includes("import"))).toBe(true);  // use_declaration -> @import
    expect(types.some(t => t.includes("struct"))).toBe(true);
    expect(types.some(t => t.includes("impl"))).toBe(true);
    expect(types.some(t => t.includes("trait"))).toBe(true);
    expect(types.some(t => t.includes("enum"))).toBe(true);
    expect(types.some(t => t.includes("func"))).toBe(true);
  });

  test("struct, impl, and trait all score 100", async () => {
    const points = await getASTBreakPoints(RS_SAMPLE, "auth.rs");
    const structPoint = points.find(p => p.type === "ast:struct");
    const implPoint = points.find(p => p.type === "ast:impl");
    const traitPoint = points.find(p => p.type === "ast:trait");

    expect(structPoint?.score).toBe(100);
    expect(implPoint?.score).toBe(100);
    expect(traitPoint?.score).toBe(100);
  });
});

// =============================================================================
// AST Break Points - C#
// =============================================================================

describe("getASTBreakPoints - C#", () => {
  const CS_SAMPLE = `using System;
using System.Collections.Generic;

namespace Qmd.Auth;

public class AuthService
{
    public AuthService(IDictionary<string, string> config)
    {
    }

    public bool Authenticate(string token)
    {
        return token.Length > 0;
    }
}

public record AuthResult(bool Success, string? Error);
`;

  test("produces break points for using, namespace, type, constructor, and method", async () => {
    const points = await getASTBreakPoints(CS_SAMPLE, "src/AuthService.cs");
    const types = points.map(p => p.type);

    expect(types).toContain("ast:import");
    expect(types).toContain("ast:namespace");
    expect(types).toContain("ast:type");
    expect(types).toContain("ast:ctor");
    expect(types).toContain("ast:method");
  });

  test("scores align with expected hierarchy", async () => {
    const points = await getASTBreakPoints(CS_SAMPLE, "src/AuthService.cs");

    const namespacePoint = points.find(p => p.type === "ast:namespace");
    const typePoint = points.find(p => p.type === "ast:type");
    const ctorPoint = points.find(p => p.type === "ast:ctor");
    const methodPoint = points.find(p => p.type === "ast:method");
    const importPoint = points.find(p => p.type === "ast:import");

    expect(namespacePoint?.score).toBe(100);
    expect(typePoint?.score).toBe(100);
    expect(ctorPoint?.score).toBe(90);
    expect(methodPoint?.score).toBe(90);
    expect(importPoint?.score).toBe(60);
  });
});

// =============================================================================
// Error Handling & Fallback
// =============================================================================

describe("getASTBreakPoints - error handling", () => {
  test("returns empty array for unsupported file types", async () => {
    const points = await getASTBreakPoints("# Hello World", "readme.md");
    expect(points).toEqual([]);
  });

  test("returns empty array for unknown extensions", async () => {
    const points = await getASTBreakPoints("data,here", "file.csv");
    expect(points).toEqual([]);
  });

  test("handles empty content gracefully", async () => {
    const points = await getASTBreakPoints("", "empty.ts");
    expect(points).toEqual([]);
  });

  test("handles syntactically invalid code gracefully", async () => {
    // Tree-sitter is error-tolerant, so this should still parse (with error nodes)
    // but should not crash
    const points = await getASTBreakPoints("function { broken syntax %%%", "broken.ts");
    // Should either return some partial break points or empty array — not throw
    expect(Array.isArray(points)).toBe(true);
  });
});

// =============================================================================
// Symbol Extraction Stub (Phase 2)
// =============================================================================

describe("extractSymbols", () => {
  test("returns empty array (Phase 2 stub)", () => {
    const symbols = extractSymbols("function foo() {}", "typescript", 0, 18);
    expect(symbols).toEqual([]);
  });

  test("returns an array for C# content", () => {
    const symbols = extractSymbols("public class InventoryService {}", "csharp", 0, 32);
    expect(Array.isArray(symbols)).toBe(true);
  });
});
