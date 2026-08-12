/**
 * test/template-retrieval.test.ts - Tests for template-aware code retrieval engine
 */

import { describe, test, expect } from "vitest";
import {
  detectTemplateAdapter,
  getTemplateBreakPoints,
  getTemplateContext,
} from "../src/templates.js";
import { chunkDocumentAsync } from "../src/store.js";
import { detectLanguage } from "../src/ast.js";

// =============================================================================
// File Detection & Priority
// =============================================================================

describe("Template Engine Detection", () => {
  test("detects Blade templates", () => {
    const adapter = detectTemplateAdapter("resources/views/orders/show.blade.php");
    expect(adapter).not.toBeNull();
    expect(adapter?.engine).toBe("blade");
    // Standard PHP language detector must return null for .blade.php
    expect(detectLanguage("resources/views/orders/show.blade.php")).toBeNull();
  });

  test("detects Twig templates", () => {
    const adapter1 = detectTemplateAdapter("templates/orders/show.twig");
    expect(adapter1?.engine).toBe("twig");

    const adapter2 = detectTemplateAdapter("templates/orders/show.html.twig");
    expect(adapter2?.engine).toBe("twig");
  });

  test("detects Smarty templates", () => {
    const adapter1 = detectTemplateAdapter("templates/orders/show.tpl");
    expect(adapter1?.engine).toBe("smarty");

    const adapter2 = detectTemplateAdapter("templates/orders/show.smarty");
    expect(adapter2?.engine).toBe("smarty");
  });

  test("detects Latte templates", () => {
    const adapter = detectTemplateAdapter("templates/orders/show.latte");
    expect(adapter?.engine).toBe("latte");
  });

  test("returns null for non-template files", () => {
    expect(detectTemplateAdapter("app/Services/OrderService.php")).toBeNull();
    expect(detectTemplateAdapter("src/auth.ts")).toBeNull();
    expect(detectTemplateAdapter("docs/README.md")).toBeNull();
  });
});

// =============================================================================
// Structural Breakpoints & Context - Blade
// =============================================================================

describe("Blade Template Breakpoints & Context", () => {
  const BLADE_SAMPLE = `@extends('layouts.app')

@section('content')
<div class="order-details">
    <x-order-card :order="$order" />
    <livewire:payment-status :id="$order->id" />

    @if($order->isPaid())
        <p>Order paid: {{ $order->customer->name }}</p>
    @else
        <p>Payment pending</p>
    @endif
</div>
@endsection
`;

  test("extracts Blade breakpoints for directives and components", () => {
    const points = getTemplateBreakPoints(BLADE_SAMPLE, "resources/views/orders/show.blade.php");
    expect(points.length).toBeGreaterThan(0);

    const types = points.map(p => p.type);
    expect(types.some(t => t.includes("extends"))).toBe(true);
    expect(types.some(t => t.includes("section"))).toBe(true);
    expect(types.some(t => t.includes("x-component"))).toBe(true);
    expect(types.some(t => t.includes("livewire"))).toBe(true);
    expect(types.some(t => t.includes("if"))).toBe(true);
  });

  test("extracts Blade context (extends layout, sections, components)", () => {
    const ctx = getTemplateContext(BLADE_SAMPLE, "resources/views/orders/show.blade.php");
    expect(ctx).not.toBeNull();
    expect(ctx?.engine).toBe("blade");
    expect(ctx?.extendsLayout).toBe("layouts.app");
    expect(ctx?.sections).toContain("content");
    expect(ctx?.components).toContain("x-order-card");
    expect(ctx?.components).toContain("livewire:payment-status");
  });
});

// =============================================================================
// Structural Breakpoints & Context - Twig
// =============================================================================

describe("Twig Template Breakpoints & Context", () => {
  const TWIG_SAMPLE = `{% extends "base.html.twig" %}

{% block content %}
    <h1>Order #{{ order.id }}</h1>
    {% if order.isPaid %}
        <p>Customer: {{ order.customer.name }}</p>
    {% endif %}
{% endblock %}
`;

  test("extracts Twig breakpoints and context", () => {
    const points = getTemplateBreakPoints(TWIG_SAMPLE, "templates/orders/show.html.twig");
    expect(points.length).toBeGreaterThan(0);

    const ctx = getTemplateContext(TWIG_SAMPLE, "templates/orders/show.html.twig");
    expect(ctx?.engine).toBe("twig");
    expect(ctx?.extendsLayout).toBe("base.html.twig");
    expect(ctx?.sections).toContain("content");
  });
});

// =============================================================================
// Structural Breakpoints & Context - Smarty & Latte
// =============================================================================

describe("Smarty & Latte Template Engine Support", () => {
  test("extracts Smarty context without file= attribute", () => {
    const SMARTY_ALT = `{extends 'layout_alt.tpl'}
{block 'content'}
    <p>Alt Smarty</p>
{/block}`;
    const ctx = getTemplateContext(SMARTY_ALT, "show_alt.tpl");
    expect(ctx?.engine).toBe("smarty");
    expect(ctx?.extendsLayout).toBe("layout_alt.tpl");
    expect(ctx?.sections).toContain("content");
  });

  test("handles non-template files in getTemplateBreakPoints and getTemplateContext", () => {
    expect(getTemplateBreakPoints("content", "file.txt")).toEqual([]);
    expect(getTemplateContext("content", "file.txt")).toBeNull();
  });

  test("handles throwing adapters in getTemplateBreakPoints and getTemplateContext gracefully", async () => {
    const { getTemplateBreakPoints, getTemplateContext } = await import("../src/templates.js");
    expect(getTemplateBreakPoints("content", "nonexistent.blade.php")).toBeDefined();
    expect(getTemplateContext("content", "nonexistent.blade.php")).toBeDefined();
  });
});

// =============================================================================
// Integration with chunkDocumentAsync
// =============================================================================

describe("Template Chunk Integration", () => {
  test("chunkDocumentAsync enriches template chunks in auto mode", async () => {
    const BLADE = `@extends('layouts.app')
@section('main')
<div>{{ $order->name }}</div>
@endsection`;

    const chunks = await chunkDocumentAsync(
      BLADE,
      50,
      10,
      10,
      "resources/views/order.blade.php",
      "auto",
    );

    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks[0]!.text).toContain("[Template Context:");
    expect(chunks[0]!.text).toContain("engine blade");
    expect(chunks[0]!.text).toContain("extends layouts.app");
  });
});
