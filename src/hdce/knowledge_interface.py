import json
import os
import random

class BloomFilter:
    def __init__(self, size=10000, hash_funcs=None):
        # Bloom Filterのためのビット配列
        self.size = size
        self.bit_array = [0]*size
        if hash_funcs is None:
            # デフォルトのハッシュ関数2つ
            self.hash_funcs = [
                lambda x: hash("1"+x) % size,
                lambda x: hash("2"+x) % size
            ]
        else:
            self.hash_funcs = hash_funcs

    def add(self, item):
        # アイテムをBloom Filterに追加
        for hf in self.hash_funcs:
            self.bit_array[hf(item)] = 1

    def might_contain(self, item):
        # すべてのビットが1なら含まれる可能性がある
        for hf in self.hash_funcs:
            if self.bit_array[hf(item)] == 0:
                return False
        return True

class KnowledgeInterface:
    def __init__(self, knowledge_graph_file, bloom_filter=None):
        self.bloom_filter = bloom_filter
        # knowledge_graph_fileがあれば読み込む
        if os.path.isfile(knowledge_graph_file):
            with open(knowledge_graph_file, "r") as f:
                self.knowledge_graph = json.load(f)
        else:
            self.knowledge_graph = {}

    def query_candidates(self, concept):
        # Bloom Filterがある場合、候補にないと判定されたら空リスト
        if self.bloom_filter is not None:
            if not self.bloom_filter.might_contain(concept):
                return []
        # 指定したconceptに対応する要素を返す
        return self.knowledge_graph.get(concept, [])

    def probabilistic_graph_walk(self, start_node, steps=2, branching=2):
        # ランダムウォーク的にグラフを探索
        results = []
        current = [start_node]
        for _ in range(steps):
            new_front = []
            for c in current:
                neighbors = self.knowledge_graph.get(c, [])
                if not neighbors:
                    continue
                # branchingの数だけランダムサンプリング
                sampled = random.sample(neighbors, min(branching, len(neighbors)))
                for (nbr, rel, pl) in sampled:
                    results.append((c, nbr, rel, pl))
                    new_front.append(nbr)
            current = new_front
        return results 