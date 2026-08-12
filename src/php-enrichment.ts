/**
 * PHP AST Content Enrichment module for QMD.
 *
 * Extracts framework-agnostic structural context (namespace, class, interface, trait,
 * enum, inheritance, attributes, and PHPDoc) from Tree-sitter PHP AST nodes.
 */

export interface PHPDeclarationContext {
  namespace?: string;
  className?: string;
  interfaceName?: string;
  traitName?: string;
  enumName?: string;
  extendsClass?: string;
  implementsInterfaces?: string[];
  usedTraits?: string[];
  methodName?: string;
  methodVisibility?: string;
  isStatic?: boolean;
  functionName?: string;
  phpDoc?: string;
  attributes?: string[];
}

/**
 * Format a PHP structural context object into a concise header string to enrich chunk content.
 */
export function formatPHPContext(ctx: PHPDeclarationContext): string {
  const parts: string[] = [];

  if (ctx.namespace) {
    parts.push(`namespace ${ctx.namespace}`);
  }

  if (ctx.className) {
    let decl = `class ${ctx.className}`;
    if (ctx.extendsClass) decl += ` extends ${ctx.extendsClass}`;
    if (ctx.implementsInterfaces && ctx.implementsInterfaces.length > 0) {
      decl += ` implements ${ctx.implementsInterfaces.join(", ")}`;
    }
    parts.push(decl);
  } else if (ctx.interfaceName) {
    let decl = `interface ${ctx.interfaceName}`;
    if (ctx.extendsClass) decl += ` extends ${ctx.extendsClass}`;
    parts.push(decl);
  } else if (ctx.traitName) {
    parts.push(`trait ${ctx.traitName}`);
  } else if (ctx.enumName) {
    parts.push(`enum ${ctx.enumName}`);
  }

  if (ctx.usedTraits && ctx.usedTraits.length > 0) {
    parts.push(`uses [${ctx.usedTraits.join(", ")}]`);
  }

  if (ctx.methodName) {
    const vis = ctx.methodVisibility ? `${ctx.methodVisibility} ` : "";
    const st = ctx.isStatic ? "static " : "";
    parts.push(`method ${vis}${st}${ctx.methodName}`);
  } else if (ctx.functionName) {
    parts.push(`function ${ctx.functionName}`);
  }

  if (ctx.attributes && ctx.attributes.length > 0) {
    parts.push(`attributes [${ctx.attributes.join(", ")}]`);
  }

  if (ctx.phpDoc) {
    // Keep single-line sanitized summary of doc block
    const cleanDoc = ctx.phpDoc.replace(/\/\*\*|\*\/|\*/g, " ").replace(/\s+/g, " ").trim();
    if (cleanDoc) {
      parts.push(`doc: ${cleanDoc}`);
    }
  }

  if (parts.length === 0) return "";
  return `// Context: ${parts.join(" | ")}`;
}

/**
 * Extract AST-based PHP structural context for specific positions in a file.
 */
export function extractPHPContextFromAST(
  rootNode: any,
  content: string,
  targetPos: number,
): PHPDeclarationContext {
  const ctx: PHPDeclarationContext = {};

  function findAncestorOrCurrent(node: any, types: string[]) {
    let curr = node;
    while (curr) {
      if (types.includes(curr.type)) return curr;
      curr = curr.parent;
    }
    return null;
  }

  // Find node at target position
  let node = rootNode.descendantForIndex(targetPos);
  if (!node) return ctx;

  // 1. Namespace
  let curr: any = node;
  while (curr) {
    const nsNode = curr.children?.find((c: any) => c.type === "namespace_definition");
    if (nsNode) {
      const nameNode = nsNode.children?.find((c: any) => c.type === "namespace_name");
      if (nameNode) {
        ctx.namespace = content.slice(nameNode.startIndex, nameNode.endIndex).trim();
      }
      break;
    }
    curr = curr.parent;
  }

  // 2. Containing class / interface / trait / enum
  const typeNode = findAncestorOrCurrent(node, [
    "class_declaration",
    "interface_declaration",
    "trait_declaration",
    "enum_declaration",
  ]);

  if (typeNode) {
    const nameNode = typeNode.children?.find((c: any) => c.type === "name");
    if (nameNode) {
      const name = content.slice(nameNode.startIndex, nameNode.endIndex).trim();
      if (typeNode.type === "class_declaration") ctx.className = name;
      else if (typeNode.type === "interface_declaration") ctx.interfaceName = name;
      else if (typeNode.type === "trait_declaration") ctx.traitName = name;
      else if (typeNode.type === "enum_declaration") ctx.enumName = name;
    }

    // Extends
    const baseClause = typeNode.children?.find((c: any) => c.type === "base_clause");
    if (baseClause) {
      const baseNameNode = baseClause.children?.find((c: any) => c.type === "name" || c.type === "qualified_name");
      if (baseNameNode) {
        ctx.extendsClass = content.slice(baseNameNode.startIndex, baseNameNode.endIndex).trim();
      }
    }

    // Implements
    const ifaceClause = typeNode.children?.find((c: any) => c.type === "class_interface_clause");
    if (ifaceClause) {
      const ifaces: string[] = [];
      for (const child of ifaceClause.children || []) {
        if (child.type === "name" || child.type === "qualified_name") {
          ifaces.push(content.slice(child.startIndex, child.endIndex).trim());
        }
      }
      if (ifaces.length > 0) ctx.implementsInterfaces = ifaces;
    }

    // Used traits inside class/enum/trait
    const bodyNode = typeNode.children?.find((c: any) => c.type === "declaration_list");
    if (bodyNode) {
      const traits: string[] = [];
      for (const child of bodyNode.children || []) {
        if (child.type === "use_declaration") {
          const traitNames = child.children?.filter((c: any) => c.type === "name" || c.type === "qualified_name");
          for (const tn of traitNames || []) {
            traits.push(content.slice(tn.startIndex, tn.endIndex).trim());
          }
        }
      }
      if (traits.length > 0) ctx.usedTraits = traits;
    }
  }

  // 3. Method or Function declaration
  const methodOrFuncNode = findAncestorOrCurrent(node, ["method_declaration", "function_definition"]);
  if (methodOrFuncNode) {
    const nameNode = methodOrFuncNode.children?.find((c: any) => c.type === "name");
    if (nameNode) {
      const name = content.slice(nameNode.startIndex, nameNode.endIndex).trim();
      if (methodOrFuncNode.type === "method_declaration") {
        ctx.methodName = name;
        // Visibility
        const visNode = methodOrFuncNode.children?.find((c: any) =>
          ["public", "protected", "private"].includes(content.slice(c.startIndex, c.endIndex)),
        );
        if (visNode) ctx.methodVisibility = content.slice(visNode.startIndex, visNode.endIndex);
        const staticNode = methodOrFuncNode.children?.find((c: any) =>
          content.slice(c.startIndex, c.endIndex) === "static",
        );
        if (staticNode) ctx.isStatic = true;
      } else {
        ctx.functionName = name;
      }
    }

    // Associated Attributes
    const attrList = methodOrFuncNode.children?.find((c: any) => c.type === "attribute_list");
    if (attrList) {
      const attrs: string[] = [];
      for (const group of attrList.children || []) {
        if (group.type === "attribute_group") {
          for (const attr of group.children || []) {
            if (attr.type === "attribute") {
              attrs.push(content.slice(attr.startIndex, attr.endIndex).trim());
            }
          }
        }
      }
      if (attrs.length > 0) ctx.attributes = attrs;
    }

    // Immediate PHPDoc comment before declaration
    const prevSibling = methodOrFuncNode.previousSibling;
    if (prevSibling && prevSibling.type === "comment") {
      const commentText = content.slice(prevSibling.startIndex, prevSibling.endIndex);
      if (commentText.startsWith("/**")) {
        ctx.phpDoc = commentText;
      }
    }
  }

  return ctx;
}
