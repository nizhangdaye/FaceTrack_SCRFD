class HungarianAlgorithm:
    def __init__(self):
        pass

    def __del__(self):
        pass

    def solve(self, dist_matrix, assignment):
        n_rows = len(dist_matrix)
        n_cols = len(dist_matrix[0])

        dist_matrix_in = [0] * (n_rows * n_cols)
        assignment_array = [-1] * n_rows
        cost = 0.0

        # 填充 dist_matrix_in
        for i in range(n_rows):
            for j in range(n_cols):
                dist_matrix_in[i + n_rows * j] = dist_matrix[i][j]

        # 调用解决函数
        self.assignment_optimal(assignment_array, cost, dist_matrix_in, n_rows, n_cols)

        assignment.clear()
        for r in range(n_rows):
            assignment.append(assignment_array[r])

        return cost

    def assignment_optimal(self, assignment, cost, dist_matrix_in, n_of_rows, n_of_columns):
        dist_matrix = [0] * (n_of_rows * n_of_columns)
        covered_columns = [False] * n_of_columns
        covered_rows = [False] * n_of_rows
        star_matrix = [False] * (n_of_rows * n_of_columns)
        prime_matrix = [False] * (n_of_rows * n_of_columns)
        new_star_matrix = [False] * (n_of_rows * n_of_columns)

        # 初始化
        cost[0] = 0
        for row in range(n_of_rows):
            assignment[row] = -1

        # 生成距离矩阵的工作副本
        n_of_elements = n_of_rows * n_of_columns
        for row in range(n_of_elements):
            value = dist_matrix_in[row]
            if value < 0:
                raise ValueError("All matrix elements have to be non-negative.")
            dist_matrix[row] = value

        # 预备步骤
        if n_of_rows <= n_of_columns:
            min_dim = n_of_rows

            for row in range(n_of_rows):
                # 找到行中最小的元素
                min_value = min(dist_matrix[row:n_of_elements:n_of_rows])

                # 从每个元素中减去最小元素
                for j in range(row, n_of_elements, n_of_rows):
                    dist_matrix[j] -= min_value

            # 步骤 1 和 2a
            for row in range(n_of_rows):
                for col in range(n_of_columns):
                    if abs(dist_matrix[row + n_of_rows * col]) < 1e-10 and not covered_columns[col]:
                        star_matrix[row + n_of_rows * col] = True
                        covered_columns[col] = True
                        break
        else:
            min_dim = n_of_columns

            for col in range(n_of_columns):
                # 找到列中最小的元素
                min_value = min(dist_matrix[n_of_rows * col:n_of_rows * col + n_of_rows])

                # 从每个元素中减去最小元素
                for j in range(n_of_rows):
                    dist_matrix[j + n_of_rows * col] -= min_value

            # 步骤 1 和 2a
            for col in range(n_of_columns):
                for row in range(n_of_rows):
                    if abs(dist_matrix[row + n_of_rows * col]) < 1e-10 and not covered_rows[row]:
                        star_matrix[row + n_of_rows * col] = True
                        covered_columns[col] = True
                        covered_rows[row] = True
                        break

            for row in range(n_of_rows):
                covered_rows[row] = False

        # 移动到步骤 2b
        self.step2b(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
                    n_of_rows, n_of_columns, min_dim)

        # 计算成本并移除无效分配
        self.compute_assignment_cost(assignment, cost, dist_matrix_in, n_of_rows)

    def build_assignment_vector(self, assignment, star_matrix, n_of_rows, n_of_columns):
        for row in range(n_of_rows):
            for col in range(n_of_columns):
                if star_matrix[row + n_of_rows * col]:
                    assignment[row] = col
                    break

    def compute_assignment_cost(self, assignment, cost, dist_matrix, n_of_rows):
        for row in range(n_of_rows):
            col = assignment[row]
            if col >= 0:
                cost[0] += dist_matrix[row + n_of_rows * col]

    def step2a(self, assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
               n_of_rows, n_of_columns, min_dim):
        for col in range(n_of_columns):
            for row in range(n_of_rows):
                if star_matrix[row + n_of_rows * col]:
                    covered_columns[col] = True
                    break

        # 移动到步骤 3
        self.step2b(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
                    n_of_rows, n_of_columns, min_dim)

    def step2b(self, assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
               n_of_rows, n_of_columns, min_dim):
        n_of_covered_columns = sum(covered_columns)

        if n_of_covered_columns == min_dim:
            self.build_assignment_vector(assignment, star_matrix, n_of_rows, n_of_columns)
        else:
            self.step3(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns,
                       covered_rows, n_of_rows, n_of_columns, min_dim)

    def step3(self, assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
              n_of_rows, n_of_columns, min_dim):
        zeros_found = True
        while zeros_found:
            zeros_found = False
            for col in range(n_of_columns):
                if not covered_columns[col]:
                    for row in range(n_of_rows):
                        if not covered_rows[row] and abs(dist_matrix[row + n_of_rows * col]) < 1e-10:
                            prime_matrix[row + n_of_rows * col] = True
                            for star_col in range(n_of_columns):
                                if star_matrix[row + n_of_rows * star_col]:
                                    break

                            if star_col == n_of_columns:
                                self.step4(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix,
                                           covered_columns, covered_rows, n_of_rows, n_of_columns, min_dim, row, col)
                                return
                            else:
                                covered_rows[row] = True
                                covered_columns[star_col] = False
                                zeros_found = True
                                break

        # 移动到步骤 5
        self.step5(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
                   n_of_rows, n_of_columns, min_dim)

    def step4(self, assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
              n_of_rows, n_of_columns, min_dim, row, col):
        # 生成 starMatrix 的临时副本
        new_star_matrix[:] = star_matrix[:]

        # 当前零的星标
        new_star_matrix[row + n_of_rows * col] = True

        # 在当前列中查找已星标零
        star_col = col
        for star_row in range(n_of_rows):
            if star_matrix[star_row + n_of_rows * star_col]:
                break

        while star_row < n_of_rows:
            new_star_matrix[star_row + n_of_rows * star_col] = False
            prime_row = star_row

            for prime_col in range(n_of_columns):
                if prime_matrix[prime_row + n_of_rows * prime_col]:
                    break

            new_star_matrix[prime_row + n_of_rows * prime_col] = True

            star_col = prime_col
            for star_row in range(n_of_rows):
                if star_matrix[star_row + n_of_rows * star_col]:
                    break

        # 用临时副本作为新的 starMatrix
        prime_matrix[:] = [False] * (n_of_rows * n_of_columns)
        star_matrix[:] = new_star_matrix[:]
        for n in range(n_of_rows):
            covered_rows[n] = False

        self.step2a(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
                    n_of_rows, n_of_columns, min_dim)

    def step5(self, assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
              n_of_rows, n_of_columns, min_dim):
        h = float('inf')

        for row in range(n_of_rows):
            if not covered_rows[row]:
                for col in range(n_of_columns):
                    if not covered_columns[col]:
                        value = dist_matrix[row + n_of_rows * col]
                        if value < h:
                            h = value

        # 在每个覆盖的行上加 h
        for row in range(n_of_rows):
            if covered_rows[row]:
                for col in range(n_of_columns):
                    dist_matrix[row + n_of_rows * col] += h

        # 从每个未覆盖的列中减去 h
        for col in range(n_of_columns):
            if not covered_columns[col]:
                for row in range(n_of_rows):
                    dist_matrix[row + n_of_rows * col] -= h

        self.step3(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
                   n_of_rows, n_of_columns, min_dim)
