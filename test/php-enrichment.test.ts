/**
 * test/php-enrichment.test.ts - Tests for PHP AST content enrichment
 */

import { describe, test, expect } from "vitest";
import { formatPHPContext, extractPHPContextFromAST } from "../src/php-enrichment.js";
import { chunkDocumentAsync } from "../src/store.js";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

describe("PHP AST Content Enrichment - Unit", () => {
  test("formats PHP context into header string", () => {
    const formatted = formatPHPContext({
      namespace: "App\\Services",
      className: "PaymentService",
      extendsClass: "BaseService",
      implementsInterfaces: ["PaymentGateway"],
      usedTraits: ["HasPayments"],
      methodName: "refund",
      methodVisibility: "public",
      attributes: ["Route('/refund')"],
      phpDoc: "/** Refund an order. */",
    });

    expect(formatted).toContain("namespace App\\Services");
    expect(formatted).toContain("class PaymentService extends BaseService implements PaymentGateway");
    expect(formatted).toContain("uses [HasPayments]");
    expect(formatted).toContain("method public refund");
    expect(formatted).toContain("attributes [Route('/refund')]");
    expect(formatted).toContain("doc: Refund an order.");
  });

  test("extracts PHP context for interfaces, traits, enums, functions, and static methods", async () => {
    const mod = await import("web-tree-sitter");
    await mod.Parser.init();
    const wasmPath = require.resolve("tree-sitter-php/tree-sitter-php.wasm");
    const lang = await mod.Language.load(wasmPath);
    const parser = new mod.Parser();
    parser.setLanguage(lang);

    const sample = `<?php

namespace App\\Contracts;

interface PaymentGateway extends BaseGateway
{
    public static function charge(): void;
}

trait HasPayments
{
    public function processPayment() {}
}

enum PaymentStatus: string
{
    case Pending = 'pending';
}

function normalize_order(array $data): array
{
    return $data;
}
`;

    const tree = parser.parse(sample);

    // Interface
    const ifacePos = sample.indexOf("interface PaymentGateway");
    const ifaceCtx = extractPHPContextFromAST(tree.rootNode, sample, ifacePos);
    expect(ifaceCtx.interfaceName).toBe("PaymentGateway");
    expect(ifaceCtx.extendsClass).toBe("BaseGateway");
    expect(formatPHPContext(ifaceCtx)).toContain("interface PaymentGateway extends BaseGateway");

    // Static Method inside Interface
    const staticPos = sample.indexOf("public static function charge");
    const staticCtx = extractPHPContextFromAST(tree.rootNode, sample, staticPos);
    expect(staticCtx.isStatic).toBe(true);
    expect(formatPHPContext(staticCtx)).toContain("method public static charge");

    // Trait
    const traitPos = sample.indexOf("trait HasPayments");
    const traitCtx = extractPHPContextFromAST(tree.rootNode, sample, traitPos);
    expect(traitCtx.traitName).toBe("HasPayments");
    expect(formatPHPContext(traitCtx)).toContain("trait HasPayments");

    // Enum
    const enumPos = sample.indexOf("enum PaymentStatus");
    const enumCtx = extractPHPContextFromAST(tree.rootNode, sample, enumPos);
    expect(enumCtx.enumName).toBe("PaymentStatus");
    expect(formatPHPContext(enumCtx)).toContain("enum PaymentStatus");

    // Global Function
    const funcPos = sample.indexOf("function normalize_order");
    const funcCtx = extractPHPContextFromAST(tree.rootNode, sample, funcPos);
    expect(funcCtx.functionName).toBe("normalize_order");
    expect(formatPHPContext(funcCtx)).toContain("function normalize_order");

    tree.delete();
    parser.delete();
  });

  test("formatPHPContext returns empty string for empty context", () => {
    expect(formatPHPContext({})).toBe("");
  });
});

describe("PHP Chunk Content Enrichment Integration", () => {
  const LARAVEL_SERVICE_SAMPLE = `<?php

namespace App\\Services;

use App\\Models\\Order;

class OrderService
{
    public function createOrder(array $data): Order
    {
        return new Order($data);
    }

    public function cancelOrder(int $orderId): bool
    {
        return true;
    }
}
`;

  test("chunkDocumentAsync enriches PHP chunks in auto mode", async () => {
    const chunks = await chunkDocumentAsync(
      LARAVEL_SERVICE_SAMPLE,
      100,
      10,
      10,
      "app/Services/OrderService.php",
      "auto",
    );

    expect(chunks.length).toBeGreaterThan(0);
    const firstChunk = chunks[0]!;
    expect(firstChunk.text).toContain("// Context:");
    expect(firstChunk.text).toContain("OrderService");
  });

  test("chunkDocumentAsync skips enrichment when strategy is regex", async () => {
    const chunks = await chunkDocumentAsync(
      LARAVEL_SERVICE_SAMPLE,
      100,
      10,
      10,
      "app/Services/OrderService.php",
      "regex",
    );

    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks[0]!.text).not.toContain("// Context:");
  });
});
