# API Design Principles

## Introduction

Good API design is crucial for developer experience. This document outlines the core principles we follow when designing REST APIs.

## Principle 1: Use Nouns, Not Verbs

URLs should represent resources, not actions. Use HTTP methods to indicate the action.

**Good:**
- GET /users/123
- POST /orders
- DELETE /products/456

**Bad:**
- GET /getUser?id=123
- POST /createOrder
- GET /deleteProduct/456

## Principle 2: Use Plural Nouns

Always use plural nouns for consistency.

- /users (not /user)
- /orders (not /order)
- /products (not /product)

## Principle 3: Hierarchical Relationships

Express relationships through URL hierarchy.

- GET /users/123/orders - Get all orders for user 123
- GET /users/123/orders/456 - Get specific order 456 for user 123

## Principle 4: Filtering and Pagination

Use query parameters for filtering, sorting, and pagination.

- GET /products?category=electronics&sort=price&page=2&limit=20

## Principle 5: Versioning

Always version your APIs. We prefer URL versioning.

- /v1/users
- /v2/users

## Principle 6: Error Handling

Return consistent error responses with appropriate HTTP status codes.

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Email format is invalid",
    "field": "email"
  }
}
```

## Principle 7: Rate Limiting

Implement rate limiting and communicate limits via headers:

- X-RateLimit-Limit: 1000
- X-RateLimit-Remaining: 999
- X-RateLimit-Reset: 1640000000

## Conclusion

Following these principles leads to APIs that are intuitive, consistent, and easy to maintain. Remember: the best API is one that developers can use without reading documentation.
