/**
 * ast.test.ts - Tests for AST-aware chunking support
 *
 * Tests language detection, AST break point extraction for each
 * supported language, and graceful fallback on errors.
 */

import { describe, test, expect } from "vitest";
import { detectLanguage, getASTBreakPoints, extractSymbols, extractAllSymbols, parseCodeFile } from "../src/ast.js";
import type { SupportedLanguage, InternalSymbol, SymbolInfo } from "../src/ast.js";

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
// Symbol Extraction
// =============================================================================

describe("extractAllSymbols - TypeScript", () => {
  const TS_CODE = `import { Database } from './db';

interface AuthConfig {
  secret: string;
}

type UserId = string;

export class AuthService {
  constructor(private db: Database) {}

  async authenticate(user: User, token: string): Promise<boolean> {
    return true;
  }

  validateToken(token: string): boolean {
    return token.length === 64;
  }
}

export function hashPassword(password: string): string {
  return 'hash';
}

enum Role {
  Admin = 'admin',
  User = 'user',
}
`;

  test("extracts class, function, method, interface, type, and enum symbols", async () => {
    const symbols = await extractAllSymbols(TS_CODE, "auth.ts");
    const names = symbols.map(s => s.name);

    expect(names).toContain("AuthConfig");
    expect(names).toContain("AuthService");
    expect(names).toContain("authenticate");
    expect(names).toContain("validateToken");
    expect(names).toContain("hashPassword");
    expect(names).toContain("constructor");
    expect(names).toContain("UserId");
    expect(names).toContain("Role");
  });

  test("assigns correct kinds", async () => {
    const symbols = await extractAllSymbols(TS_CODE, "auth.ts");
    const byName = Object.fromEntries(symbols.map(s => [s.name, s]));

    expect(byName.AuthConfig?.kind).toBe("interface");
    expect(byName.AuthService?.kind).toBe("class");
    expect(byName.authenticate?.kind).toBe("method");
    expect(byName.hashPassword?.kind).toBe("function");
    expect(byName.UserId?.kind).toBe("type");
    expect(byName.Role?.kind).toBe("enum");
  });

  test("extracts signatures with parameters and return types", async () => {
    const symbols = await extractAllSymbols(TS_CODE, "auth.ts");
    const byName = Object.fromEntries(symbols.map(s => [s.name, s]));

    expect(byName.hashPassword?.signature).toContain("(password: string)");
    expect(byName.hashPassword?.signature).toContain(": string");
    expect(byName.authenticate?.signature).toContain("(user: User, token: string)");
    expect(byName.authenticate?.signature).toContain(": Promise<boolean>");
  });

  test("includes line numbers", async () => {
    const symbols = await extractAllSymbols(TS_CODE, "auth.ts");
    for (const s of symbols) {
      expect(s.line).toBeGreaterThan(0);
    }
  });

  test("includes byte offsets (pos)", async () => {
    const symbols = await extractAllSymbols(TS_CODE, "auth.ts");
    for (const s of symbols) {
      expect(s.pos).toBeGreaterThanOrEqual(0);
    }
    // Symbols should be sorted by position
    for (let i = 1; i < symbols.length; i++) {
      expect(symbols[i]!.pos).toBeGreaterThanOrEqual(symbols[i - 1]!.pos);
    }
  });

  test("handles unicode identifiers", async () => {
    const code = `
export class Über {
  método(données: string): void {}
}

function 计算(値: number): number { return 値; }
`;
    const symbols = await extractAllSymbols(code, "unicode.ts");
    const names = symbols.map(s => s.name);
    expect(names).toContain("Über");
    expect(names).toContain("método");
    expect(names).toContain("计算");
  });
});

// JavaScript uses the TypeScript grammar internally (see detectLanguage → ast.ts).
// Tests use a .js extension to verify that the TS parser handles plain JS files.
describe("extractAllSymbols - JavaScript", () => {
  const JS_CODE = `import { db } from './db';

class UserService {
  constructor(db) {
    this.db = db;
  }

  async findUser(id) {
    return await this.db.find(id);
  }
}

function createUser(name, email) {
  return { name, email };
}

const deleteUser = (id) => {
  return db.remove(id);
};

export default function mainHandler(req, res) {
  res.send('ok');
}
`;

  test("extracts class, function, and method symbols", async () => {
    const symbols = await extractAllSymbols(JS_CODE, "service.js");
    const names = symbols.map(s => s.name);

    expect(names).toContain("UserService");
    expect(names).toContain("constructor");
    expect(names).toContain("findUser");
    expect(names).toContain("createUser");
    expect(names).toContain("mainHandler");
  });

  test("assigns correct kinds", async () => {
    const symbols = await extractAllSymbols(JS_CODE, "service.js");
    const byName = Object.fromEntries(symbols.map(s => [s.name, s]));

    expect(byName.UserService?.kind).toBe("class");
    expect(byName.findUser?.kind).toBe("method");
    expect(byName.createUser?.kind).toBe("function");
    expect(byName.mainHandler?.kind).toBe("function");
  });

  test("handles arrow functions and default exports", async () => {
    const symbols = await extractAllSymbols(JS_CODE, "service.js");
    const names = symbols.map(s => s.name);

    // Arrow function assigned to const
    expect(names).toContain("deleteUser");

    // Default-exported function
    expect(names).toContain("mainHandler");
  });

  test("includes line numbers", async () => {
    const symbols = await extractAllSymbols(JS_CODE, "service.js");
    for (const s of symbols) {
      expect(s.line).toBeGreaterThan(0);
    }
  });

  test("includes byte offsets (pos)", async () => {
    const symbols = await extractAllSymbols(JS_CODE, "service.js");
    for (const s of symbols) {
      expect(s.pos).toBeGreaterThanOrEqual(0);
    }
    // Symbols should be sorted by position
    for (let i = 1; i < symbols.length; i++) {
      expect(symbols[i]!.pos).toBeGreaterThanOrEqual(symbols[i - 1]!.pos);
    }
  });
});

describe("extractAllSymbols - Python", () => {
  const PY_CODE = `import os
from typing import Optional

class UserService:
    def __init__(self, db):
        self.db = db

    async def find_user(self, user_id: str) -> Optional[dict]:
        return await self.db.find(user_id)

def create_user(name: str, email: str) -> dict:
    return {"name": name, "email": email}
`;

  test("extracts class and function symbols", async () => {
    const symbols = await extractAllSymbols(PY_CODE, "service.py");
    const names = symbols.map(s => s.name);

    expect(names).toContain("UserService");
    expect(names).toContain("__init__");
    expect(names).toContain("find_user");
    expect(names).toContain("create_user");
  });

  test("assigns correct kinds", async () => {
    const symbols = await extractAllSymbols(PY_CODE, "service.py");
    const byName = Object.fromEntries(symbols.map(s => [s.name, s]));

    expect(byName.UserService?.kind).toBe("class");
    expect(byName.create_user?.kind).toBe("function");
    expect(byName.__init__?.kind).toBe("function");
  });

  test("extracts Python signatures", async () => {
    const symbols = await extractAllSymbols(PY_CODE, "service.py");
    const byName = Object.fromEntries(symbols.map(s => [s.name, s]));

    expect(byName.create_user?.signature).toContain("(name: str, email: str)");
  });
});

describe("extractAllSymbols - Go", () => {
  const GO_CODE = `package main

type Server struct {
    port int
}

func NewServer(port int) *Server {
    return &Server{port: port}
}

func (s *Server) Start() error {
    return nil
}
`;

  test("extracts type, function, and method symbols", async () => {
    const symbols = await extractAllSymbols(GO_CODE, "server.go");
    const names = symbols.map(s => s.name);

    expect(names).toContain("Server");
    expect(names).toContain("NewServer");
    expect(names).toContain("Start");
  });

  test("assigns correct kinds", async () => {
    const symbols = await extractAllSymbols(GO_CODE, "server.go");
    const byName = Object.fromEntries(symbols.map(s => [s.name, s]));

    expect(byName.Server?.kind).toBe("type");
    expect(byName.NewServer?.kind).toBe("function");
    expect(byName.Start?.kind).toBe("method");
  });
});

describe("extractAllSymbols - Rust", () => {
  const RS_CODE = `use std::io;

pub struct Config {
    port: u16,
}

impl Config {
    pub fn new(port: u16) -> Self {
        Config { port }
    }
}

pub trait Configurable {
    fn configure(&mut self);
}

pub enum ServerState {
    Running,
    Stopped,
}

pub fn start_server(port: u16) -> io::Result<()> {
    Ok(())
}
`;

  test("extracts struct, impl, trait, enum, and function symbols", async () => {
    const symbols = await extractAllSymbols(RS_CODE, "config.rs");
    const names = symbols.map(s => s.name);

    expect(names).toContain("Config");
    expect(names).toContain("Configurable");
    expect(names).toContain("ServerState");
    expect(names).toContain("start_server");
    expect(names).toContain("new");
  });

  test("assigns correct kinds", async () => {
    const symbols = await extractAllSymbols(RS_CODE, "config.rs");

    // Config appears twice: as struct and as impl
    const configStruct = symbols.find(s => s.name === "Config" && s.kind === "struct");
    const configImpl = symbols.find(s => s.name === "Config" && s.kind === "impl");
    expect(configStruct).toBeDefined();
    expect(configImpl).toBeDefined();

    expect(symbols.find(s => s.name === "Configurable")?.kind).toBe("trait");
    expect(symbols.find(s => s.name === "ServerState")?.kind).toBe("enum");
    expect(symbols.find(s => s.name === "start_server")?.kind).toBe("function");
    expect(symbols.find(s => s.name === "new")?.kind).toBe("function");
  });
});

describe("parseCodeFile", () => {
  test("returns both breakpoints and symbols from a single parse", async () => {
    const code = `export function hello(): string { return "hi"; }`;
    const result = await parseCodeFile(code, "test.ts");

    expect(result.breakPoints.length).toBeGreaterThan(0);
    expect(result.symbols.length).toBeGreaterThan(0);

    const sym = result.symbols.find(s => s.name === "hello");
    expect(sym).toBeDefined();
    expect(sym?.kind).toBe("function");
  });

  test("returns empty for unsupported files", async () => {
    const result = await parseCodeFile("# Hello", "readme.md");
    expect(result.breakPoints).toEqual([]);
    expect(result.symbols).toEqual([]);
  });

  test("returns empty for empty content", async () => {
    const result = await parseCodeFile("", "empty.ts");
    expect(result.breakPoints).toEqual([]);
    expect(result.symbols).toEqual([]);
  });
});

describe("extractSymbols (range filter)", () => {
  const CODE = `function first() { return 1; }

function second() { return 2; }

function third() { return 3; }
`;

  test("filters symbols by byte range", async () => {
    const all = await extractAllSymbols(CODE, "funcs.ts");
    expect(all.length).toBe(3);

    // Get only symbols in the first 31 bytes (first function ends at pos 30)
    const firstOnly = await extractSymbols(CODE, "funcs.ts", 0, 31);
    expect(firstOnly.length).toBe(1);
    expect(firstOnly[0]?.name).toBe("first");
  });

  test("returns public SymbolInfo without pos field", async () => {
    const symbols = await extractSymbols(CODE, "funcs.ts", 0, CODE.length);
    for (const s of symbols) {
      expect(s).toHaveProperty("name");
      expect(s).toHaveProperty("kind");
      expect(s).toHaveProperty("line");
      expect(s).not.toHaveProperty("pos");
    }
  });

  test("returns empty for unsupported files", async () => {
    const symbols = await extractSymbols("# Hello", "readme.md", 0, 100);
    expect(symbols).toEqual([]);
  });

  test("returns empty for range with no symbols", async () => {
    // Pick a range that's between functions (whitespace)
    const symbols = await extractSymbols(CODE, "funcs.ts", 30, 32);
    expect(symbols).toEqual([]);
  });
});
