/**
 * test/php-ast.test.ts - Tests for PHP Tree-sitter AST support in QMD
 */

import { describe, test, expect } from "vitest";
import { detectLanguage, getASTBreakPoints, getASTStatus } from "../src/ast.js";

// =============================================================================
// Language Detection & Blade Bypass
// =============================================================================

describe("PHP Language Detection", () => {
  test("recognizes .php extension", () => {
    expect(detectLanguage("app/Services/OrderService.php")).toBe("php");
    expect(detectLanguage("src/util.PHP")).toBe("php");
  });

  test("explicitly bypasses .blade.php from PHP parser", () => {
    expect(detectLanguage("resources/views/orders/show.blade.php")).toBeNull();
  });
});

// =============================================================================
// AST Break Points - PHP
// =============================================================================

describe("getASTBreakPoints - PHP", () => {
  const PHP_SAMPLE = `<?php

namespace App\\Services;

use App\\Models\\Order;
use App\\Contracts\\PaymentGateway;

/**
 * Class OrderService
 */
class OrderService extends BaseService implements PaymentGateway
{
    public const DEFAULT_STATUS = 'pending';
    private Order $order;

    public function create(Order $order): void
    {
    }

    public function cancel(Order $order): void
    {
    }
}

interface PaymentGateway
{
    public function charge(): void;
}

trait HasPayments
{
    public function refund(): void {}
}

enum PaymentStatus: string
{
    case Pending = 'pending';
    case Paid = 'paid';
}

function normalize_order(array $data): array
{
    return $data;
}
`;

  test("produces break points for class, interface, trait, enum, method, function, namespace, and imports", async () => {
    const points = await getASTBreakPoints(PHP_SAMPLE, "app/Services/OrderService.php");
    expect(points.length).toBeGreaterThan(0);

    const types = points.map(p => p.type);
    expect(types.some(t => t.includes("ns"))).toBe(true);
    expect(types.some(t => t.includes("import"))).toBe(true);
    expect(types.some(t => t.includes("class"))).toBe(true);
    expect(types.some(t => t.includes("iface"))).toBe(true);
    expect(types.some(t => t.includes("trait"))).toBe(true);
    expect(types.some(t => t.includes("enum"))).toBe(true);
    expect(types.some(t => t.includes("method"))).toBe(true);
    expect(types.some(t => t.includes("func"))).toBe(true);
  });

  test("break points are sorted by position", async () => {
    const points = await getASTBreakPoints(PHP_SAMPLE, "app/Services/OrderService.php");
    for (let i = 1; i < points.length; i++) {
      expect(points[i]!.pos).toBeGreaterThanOrEqual(points[i - 1]!.pos);
    }
  });

  test("scores match expectations (class/interface/trait/enum/ns = 100, method/func = 90, import = 60)", async () => {
    const points = await getASTBreakPoints(PHP_SAMPLE, "app/Services/OrderService.php");

    const nsPoint = points.find(p => p.type === "ast:ns");
    expect(nsPoint?.score).toBe(100);

    const classPoint = points.find(p => p.type === "ast:class");
    expect(classPoint?.score).toBe(100);

    const methodPoint = points.find(p => p.type === "ast:method");
    expect(methodPoint?.score).toBe(90);

    const importPoint = points.find(p => p.type === "ast:import");
    expect(importPoint?.score).toBe(60);
  });

  test("Laravel controller fixture works cleanly", async () => {
    const CONTROLLER_FIXTURE = `<?php

namespace App\\Http\\Controllers;

use App\\Services\\OrderService;

class OrderController
{
    public function store(OrderService $orders)
    {
        return $orders->create();
    }

    public function cancel(OrderService $orders)
    {
        return $orders->cancel();
    }
}
`;
    const points = await getASTBreakPoints(CONTROLLER_FIXTURE, "app/Http/Controllers/OrderController.php");
    expect(points.length).toBeGreaterThan(0);
    expect(points.some(p => p.type === "ast:class")).toBe(true);
    expect(points.filter(p => p.type === "ast:method").length).toBe(2);
  });
});

// =============================================================================
// Grammar Status / Health Check
// =============================================================================

describe("PHP Grammar Status", () => {
  test("reports php as available in getASTStatus", async () => {
    const status = await getASTStatus();
    expect(status.available).toBe(true);
    const phpLang = status.languages.find(l => l.language === "php");
    expect(phpLang).toBeDefined();
    expect(phpLang?.available).toBe(true);
  });
});
