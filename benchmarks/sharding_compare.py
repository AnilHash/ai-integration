from benchmarks.scale_benchmark import run

run("scale_100k_1shard", 100000, shard_number=1)
run("scale_100k_4shard", 100000, shard_number=4)
# $ python -m benchmarks.sharding_compare
# scale_100k_1shard: building 100000 vectors, on_disk=False, shards=1...
# {'name': 'scale_100k_1shard', 'n': 100000, 'on_disk': False, 'shards': 1, 'upload_s': 584.3, 'avg_query_ms': 23.77}
# scale_100k_4shard: building 100000 vectors, on_disk=False, shards=4...
# {'name': 'scale_100k_4shard', 'n': 100000, 'on_disk': False, 'shards': 4, 'upload_s': 384.6, 'avg_query_ms': 37.93}
#  python -m benchmarks.sharding_compare
# scale_100k_1shard: building 100000 vectors, on_disk=False, shards=1...
# {'name': 'scale_100k_1shard', 'n': 100000, 'on_disk': False, 'shards': 1, 'upload_s': 687.7, 'avg_query_ms': 26.78}
# scale_100k_4shard: building 100000 vectors, on_disk=False, shards=4...
# {'name': 'scale_100k_4shard', 'n': 100000, 'on_disk': False, 'shards': 4, 'upload_s': 400.3, 'avg_query_ms': 37.56}
