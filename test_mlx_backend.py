#!/usr/bin/env python3
"""
Simple test script for the MLX backend.

This script sends test commands to the backend and verifies responses.
"""

import subprocess
import json
import sys
import time

def send_command(process, command):
    """Send a command to the backend and get response."""
    cmd_json = json.dumps(command) + '\n'
    process.stdin.write(cmd_json)
    process.stdin.flush()
    
    response = process.stdout.readline()
    return json.loads(response)

def main():
    print("Starting MLX backend test...")
    
    # Start the backend
    try:
        process = subprocess.Popen(
            ['python3', 'mlx_backend.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait for ready signal
        print("Waiting for backend to start...")
        ready = process.stdout.readline()
        ready_data = json.loads(ready)
        print(f"✓ Backend ready: {ready_data}")
        
        # Test 1: Ping
        print("\n--- Test 1: Ping ---")
        response = send_command(process, {"command": "ping"})
        print(f"Response: {response}")
        assert response.get("success"), "Ping failed"
        print("✓ Ping test passed")
        
        # Test 2: Initialize (will download models if not cached)
        print("\n--- Test 2: Initialize ---")
        print("Note: This will download ~2GB of models on first run...")
        response = send_command(process, {"command": "initialize"})
        print(f"Response: {json.dumps(response, indent=2)}")
        
        if not response.get("success"):
            print(f"✗ Initialize failed: {response.get('error')}")
            print("\nMake sure you have:")
            print("1. Installed dependencies: pip install -r requirements.txt")
            print("2. Apple Silicon Mac")
            print("3. Internet connection for model downloads")
            process.terminate()
            return
        
        print("✓ Initialize test passed")
        
        # Test 3: Embed
        print("\n--- Test 3: Embed ---")
        response = send_command(process, {
            "command": "embed",
            "text": "Hello, world!",
            "isQuery": True
        })
        
        if response.get("success"):
            embedding = response.get("embedding", [])
            print(f"✓ Embed test passed (embedding dimension: {len(embedding)})")
            print(f"  First 5 values: {embedding[:5]}")
        else:
            print(f"✗ Embed failed: {response.get('error')}")
        
        # Test 4: Embed Batch
        print("\n--- Test 4: Embed Batch ---")
        response = send_command(process, {
            "command": "embedBatch",
            "texts": ["First text", "Second text", "Third text"],
            "isQuery": False,
            "titles": ["Title 1", "Title 2", "Title 3"]
        })
        
        if response.get("success"):
            embeddings = response.get("embeddings", [])
            print(f"✓ Embed batch test passed ({len(embeddings)} embeddings)")
        else:
            print(f"✗ Embed batch failed: {response.get('error')}")
        
        # Test 5: Rerank
        print("\n--- Test 5: Rerank ---")
        response = send_command(process, {
            "command": "rerank",
            "query": "machine learning",
            "documents": [
                {"file": "doc1.md", "text": "Machine learning is a field of AI", "title": "ML Intro"},
                {"file": "doc2.md", "text": "Cooking pasta is easy", "title": "Pasta Recipe"},
                {"file": "doc3.md", "text": "Neural networks are used in ML", "title": "Neural Nets"}
            ]
        })
        
        if response.get("success"):
            results = response.get("results", [])
            print(f"✓ Rerank test passed ({len(results)} documents ranked)")
            for i, result in enumerate(results[:3]):
                print(f"  {i+1}. {result['file']}: {result['score']:.4f}")
        else:
            print(f"✗ Rerank failed: {response.get('error')}")
        
        # Test 6: Expand Query
        print("\n--- Test 6: Expand Query ---")
        response = send_command(process, {
            "command": "expandQuery",
            "query": "python programming",
            "includeLexical": True
        })
        
        if response.get("success"):
            queryables = response.get("queryables", [])
            print(f"✓ Expand query test passed ({len(queryables)} variations)")
            for q in queryables:
                print(f"  [{q['type']}] {q['text']}")
        else:
            print(f"✗ Expand query failed: {response.get('error')}")
        
        # Shutdown
        print("\n--- Shutting down ---")
        response = send_command(process, {"command": "shutdown"})
        print(f"Response: {response}")
        
        # Wait for process to finish
        process.wait(timeout=5)
        print("\n✓ All tests completed!")
        
    except FileNotFoundError:
        print("✗ Error: mlx_backend.py not found. Make sure you're in the qmd-mlx directory.")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if process:
            process.terminate()
    
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        if process:
            process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
