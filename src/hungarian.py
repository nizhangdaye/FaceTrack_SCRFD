import numpy as np


class HungarianAlgorithm:
    def __init__(self):
        pass

    def solve(self, dist_matrix):
        """
        Solves the assignment problem using the Hungarian algorithm.

        Args:
            dist_matrix (list of lists or numpy array): The cost matrix.

        Returns:
            assignment (list): List of assigned rows to columns.
            cost (float): The total cost of the optimal assignment.
        """
        dist_matrix = np.array(dist_matrix)
        n_rows, n_cols = dist_matrix.shape

        assignment = [-1] * n_rows
        cost = 0.0

        # 将矩阵转换为行优先的扁平化格式
        dist_matrix_flat = dist_matrix.flatten()

        # Call the function to solve the assignment problem
        assignment, cost = self.assignmentoptimal(assignment, cost, dist_matrix_flat, n_rows, n_cols)

        return assignment, cost

    def assignmentoptimal(self, assignment, cost, dist_matrix_flat, n_rows, n_cols):
        # Initialize
        cost = 0
        assignment[:] = [-1] * n_rows

        # Create working copy of distance matrix
        dist_matrix = np.array(dist_matrix_flat).reshape(n_rows, n_cols)

        if np.any(dist_matrix < 0):
            print("All matrix elements have to be non-negative.")

        # Ensure all elements are non-negative
        if np.any(dist_matrix < 0):
            raise ValueError("All matrix elements have to be non-negative.")

        # Memory allocation
        covered_columns = np.zeros(n_cols, dtype=bool)
        covered_rows = np.zeros(n_rows, dtype=bool)
        star_matrix = np.zeros((n_rows, n_cols), dtype=bool)
        prime_matrix = np.zeros((n_rows, n_cols), dtype=bool)
        new_star_matrix = np.zeros((n_rows, n_cols), dtype=bool)

        # Preliminary steps
        if n_rows <= n_cols:
            min_dim = n_rows

            # Step 1a: Subtract the smallest element from each row
            for row in range(n_rows):
                min_value = np.min(dist_matrix[row, :])
                dist_matrix[row, :] -= min_value

            # Step 1b: Star assignment
            for row in range(n_rows):
                for col in range(n_cols):
                    if np.isclose(dist_matrix[row, col], 0) and not covered_columns[col]:
                        star_matrix[row, col] = True
                        covered_columns[col] = True
                        break
        else:
            min_dim = n_cols

            # Step 1a: Subtract the smallest element from each column
            for col in range(n_cols):
                min_value = np.min(dist_matrix[:, col])
                dist_matrix[:, col] -= min_value

            # Step 1b: Star assignment
            for col in range(n_cols):
                for row in range(n_rows):
                    if np.isclose(dist_matrix[row, col], 0) and not covered_rows[row]:
                        star_matrix[row, col] = True
                        covered_columns[col] = True
                        covered_rows[row] = True
                        break
            covered_rows[:] = False  # Reset covered rows

        # Move to step 2b
        self.step2b(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
                    n_rows, n_cols, min_dim)

        # Compute cost and remove invalid assignments
        self.computeassignmentcost(assignment, cost, dist_matrix_flat, n_rows)
        return assignment, cost

    def step2b(self, assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
               n_of_rows, n_of_columns, min_dim):
        while True:
            n_of_covered_columns = sum(covered_columns)
            if n_of_covered_columns == min_dim:
                self.buildassignmentvector(assignment, star_matrix, n_of_rows, n_of_columns)
                return
            self.step3(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns,
                       covered_rows, n_of_rows, n_of_columns, min_dim)

    def step3(self, assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
              n_of_rows, n_of_columns, min_dim):
        while True:
            found_zero = False
            for col in range(n_of_columns):
                if not covered_columns[col]:
                    for row in range(n_of_rows):
                        if not covered_rows[row] and np.isclose(dist_matrix[row, col], 0):
                            prime_matrix[row, col] = True
                            star_col = np.where(star_matrix[row, :])[0]
                            if star_col.size == 0:
                                self.step4(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix,
                                           covered_columns, covered_rows, n_of_rows, n_of_columns, min_dim, row, col)
                                return
                            covered_rows[row] = True
                            covered_columns[star_col[0]] = False
                            found_zero = True
                            break
                    if found_zero:
                        break
            if not found_zero:
                self.step5(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns,
                           covered_rows, n_of_rows, n_of_columns, min_dim)
                return

    def step4(self, assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
              n_of_rows, n_of_columns, min_dim, row, col):
        new_star_matrix[:] = star_matrix[:]
        new_star_matrix[row, col] = True

        while True:
            star_col = np.where(star_matrix[row, :])[0]
            if star_col.size == 0:
                break
            star_matrix[row, star_col[0]] = False
            prime_row = np.where(prime_matrix[:, star_col[0]])[0][0]
            star_matrix[prime_row, star_col[0]] = True
            row, col = prime_row, star_col[0]

        star_matrix[:] = new_star_matrix[:]
        prime_matrix[:, :] = False
        covered_rows[:] = [False] * n_of_rows
        covered_columns[:] = [False] * n_of_columns
        self.step2b(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
                    n_of_rows, n_of_columns, min_dim)

    def step5(self, assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
              n_of_rows, n_of_columns, min_dim):
        uncovered_min = np.min(dist_matrix[~np.array(covered_rows)[:, None] & ~np.array(covered_columns)[None, :]])

        dist_matrix[covered_rows, :] += uncovered_min
        dist_matrix[:, ~np.array(covered_columns)] -= uncovered_min

        self.step3(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows,
                   n_of_rows, n_of_columns, min_dim)

    def buildassignmentvector(self, assignment, star_matrix, n_of_rows, n_of_columns):
        for row in range(n_of_rows):
            star_col = np.where(star_matrix[row, :])[0]
            if star_col.size > 0:
                assignment[row] = star_col[0]

    def computeassignmentcost(self, assignment, dist_matrix, n_of_rows):
        cost = 0.0
        for row in range(n_of_rows):
            col = assignment[row]
            if col >= 0:
                cost += dist_matrix[row + n_of_rows * col]
        return cost
