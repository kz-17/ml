import random
import math

class TSPSolver:
    def __init__(self, cities, distance_matrix):
        self.cities = cities  # 城市列表 [0, 1, 2, ..., n-1]
        self.matrix = distance_matrix
        self.n = len(cities)

    def get_height(self, tour):
        """
        Height 函數：距離的負值。
        距離愈短，高度愈高。
        """
        total_dist = 0
        for i in range(self.n):
            u = tour[i]
            v = tour[(i + 1) % self.n] # 回到起點
            total_dist += self.matrix[u][v]
        return -total_dist

    def get_neighbors(self, tour):
        """
        Neighbor 函數：使用 2-opt 切換。
        選取兩個邊 (a,b) 與 (c,d)，轉換為 (a,c) 與 (b,d)。
        註：為了維持路徑連貫，這相當於反轉路徑中的一個子段。
        """
        neighbors = []
        # 隨機產生鄰居，或遍歷所有可能的 2-opt 交換
        for i in range(self.n):
            for j in range(i + 2, self.n):
                # 建立新路徑：反轉 i+1 到 j 之間的順序
                new_tour = tour[:]
                new_tour[i+1:j+1] = reversed(tour[i+1:j+1])
                neighbors.append(new_tour)
        return neighbors

    def solve(self):
        # 1. 初始解：1 => 2 => 3 => ... => n => 1
        current_tour = list(range(self.n))
        current_height = self.get_height(current_tour)
        
        print(f"初始路徑: {current_tour}")
        print(f"初始高度: {current_height}")

        while True:
            neighbors = self.get_neighbors(current_tour)
            best_neighbor = None
            max_neighbor_height = -float('inf')

            # 在所有鄰居中尋找最高點（Steepest Ascent）
            for neighbor in neighbors:
                h = self.get_height(neighbor)
                if h > max_neighbor_height:
                    max_neighbor_height = h
                    best_neighbor = neighbor

            # 如果最強鄰居比現在還高，就往上爬
            if max_neighbor_height > current_height:
                current_height = max_neighbor_height
                current_tour = best_neighbor
            else:
                # 到達局部最優點（Local Optimum），停止
                break
        
        return current_tour, -current_height
