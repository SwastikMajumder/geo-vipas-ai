import streamlit as st
from PIL import Image, ImageDraw, ImageFont  # For creating the initial black-and-white image
import re
import itertools
import math
import numpy as np
from fractions import Fraction
import copy


def ENGINE(command):
    plotter_on = True
    line_counter = []
    all_tri = []
    points = []
    point_pairs = []
    eq_list = []
    lines = []
    matrix = []
    matrix_eq = []
    line_matrix = []
    log = ""

    class TreeNode:
        def __init__(self, name, children=None):
            self.name = name
            self.children = children or []


    def tree_form(tabbed_strings):
        lines = tabbed_strings.split("\n")
        root = TreeNode("Root")
        current_level_nodes = {0: root}
        stack = [root]
        for line in lines:
            level = line.count(" ")
            node_name = line.strip()
            node = TreeNode(node_name)
            while len(stack) > level + 1:
                stack.pop()
            parent_node = stack[-1]
            parent_node.children.append(node)
            current_level_nodes[level] = node
            stack.append(node)
        return root.children[0]


    def str_form(node):
        def recursive_str(node, depth=0):
            result = "{}{}".format(" " * depth, node.name)
            for child in node.children:
                result += "\n" + recursive_str(child, depth + 1)
            return result

        return recursive_str(node)


    def line_fx(line_input):
        a = line_input[0]
        b = line_input[1]
        return TreeNode("f_line", [tree_form("d_" + a), tree_form("d_" + b)])


    def index_line(a, b):
        nonlocal line_counter
        nonlocal point_pairs
        a, b = line_sort(a + b)
        a, b = a2n(a), a2n(b)
        if (a, b) in line_counter:
            return line_counter.index((a, b))
        else:
            line_counter.append((a, b))
            fix_line_matrix()
            return len(line_counter) - 1


    def index_line_matrix(index):
        nonlocal line_counter
        a, b = line_counter[index]
        return line_sort(n2a(a) + n2a(b))


    def string_equation_helper(equation_tree):
        if equation_tree.children == []:
            return equation_tree.name
        s = "("
        if len(equation_tree.children) == 1 or equation_tree.name in [
            "f_if",
            "f_xcongruent",
            "f_congruent",
            "f_triangle",
            "f_xangle",
            "f_xline",
            "f_angle",
            "f_line",
            "f_parallel",
        ]:
            s = equation_tree.name[2:] + s
        sign = {
            "f_if": ",",
            "f_xparallel": ",",
            "f_xcongruent": ",",
            "f_congruent": ",",
            "f_triangle": "?",
            "f_add": "+",
            "f_and": "^",
            "f_dif": "?",
            "f_mul": "*",
            "f_eq": "=",
            "f_sub": "-",
            "f_angle": "?",
            "f_xangle": "?",
            "f_parallel": ",",
            "f_xline": "?",
            "f_exist": "?",
            "f_line": "?",
        }
        for child in equation_tree.children:
            s += string_equation_helper(copy.deepcopy(child)) + sign[equation_tree.name]
        s = s[:-1] + ")"
        return s


    def string_equation(eq):
        eq = eq.replace("d_", "")

        return string_equation_helper(tree_form(eq)).replace("?", "")


    def line_eq(line1, line2):
        if line1 == line2:
            return True
        line1 = a2n(line1[0]), a2n(line1[1])
        line2 = a2n(line2[0]), a2n(line2[1])
        row = [Fraction(0)] * len(line_counter)
        row[line_counter.index(line1)] = Fraction(1)
        row[line_counter.index(line2)] = Fraction(-1)
        if row in line_matrix:
            return True
        row[line_counter.index(line1)] = Fraction(-1)
        row[line_counter.index(line2)] = Fraction(1)
        if row in line_matrix:
            return True
        return False


    def angle_eq(angle1, angle2):
        if angle1 == angle2:
            return True

        row = [Fraction(0)] * len(all_angles)
        row[all_angles.index(angle1)] = Fraction(1)
        row[all_angles.index(angle2)] = Fraction(-1)
        if row in matrix and matrix_eq[matrix.index(row)] == Fraction(0):
            return True
        row[all_angles.index(angle1)] = Fraction(-1)
        row[all_angles.index(angle2)] = Fraction(1)
        if row in matrix and matrix_eq[matrix.index(row)] == Fraction(0):
            return True
        return False


    def angle_per(angle):
        row = [Fraction(0)] * len(all_angles)
        row[all_angles.index(angle)] = Fraction(1)
        if row in matrix and matrix_eq[matrix.index(row)] == Fraction(90):
            return True
        return False


    def line_counter_convert():
        output = []
        for item in line_counter:
            output.append(line_sort(n2a(item[0]) + n2a(item[1])))
        return output


    def sss_rule(a1, a2, a3, b1, b2, b3):
        line = [
            line_sort(a1 + a2),
            line_sort(b1 + b2),
            line_sort(a2 + a3),
            line_sort(b2 + b3),
            line_sort(a1 + a3),
            line_sort(b1 + b3),
        ]

        for item in line:
            if item not in line_counter_convert():
                return False

        return (
            line_eq(line[0], line[1])
            and line_eq(line[2], line[3])
            and line_eq(line[4], line[5])
        )


    def sas_rule(a1, a2, a3, b1, b2, b3):
        line = [
            line_sort(a1 + a2),
            line_sort(b1 + b2),
            line_sort(a2 + a3),
            line_sort(b2 + b3),
        ]
        angle = [plotter_angle_3(a1 + a2 + a3), plotter_angle_3(b1 + b2 + b3)]

        for item in line:
            if item not in line_counter_convert():
                return False
        for item in angle:
            if item not in all_angles:

                return False

        return (
            line_eq(line[0], line[1])
            and angle_eq(angle[0], angle[1])
            and line_eq(line[2], line[3])
        )


    def aas_rule(a1, a2, a3, b1, b2, b3):
        line = [line_sort(a2 + a3), line_sort(b2 + b3)]
        angle = [
            plotter_angle_3(a1 + a2 + a3),
            plotter_angle_3(b1 + b2 + b3),
            plotter_angle_3(a3 + a1 + a2),
            plotter_angle_3(b3 + b1 + b2),
        ]

        for item in line:
            if item not in line_counter_convert():
                return False

        for item in angle:
            if item not in all_angles:
                return False

        return (
            line_eq(line[0], line[1])
            and angle_eq(angle[0], angle[1])
            and angle_eq(angle[2], angle[3])
        )


    def rhs_rule(a1, a2, a3, b1, b2, b3):
        line = [
            line_sort(a1 + a2),
            line_sort(b1 + b2),
            line_sort(a1 + a3),
            line_sort(b1 + b3),
        ]
        angle = [plotter_angle_3(a1 + a2 + a3), plotter_angle_3(b1 + b2 + b3)]

        for item in line:
            if item not in line_counter_convert():
                return False

        for item in angle:
            if item not in all_angles:
                return False

        return (
            line_eq(line[0], line[1])
            and angle_eq(angle[0], angle[1])
            and line_eq(line[2], line[3])
            and angle_per(angle[0])
        )


    def proof_fx_3(angle1, angle2):
        nonlocal eq_list

        angle_1 = TreeNode(
            "f_triangle",
            [
                tree_form("d_" + angle1[0]),
                tree_form("d_" + angle1[1]),
                tree_form("d_" + angle1[2]),
            ],
        )
        angle_2 = TreeNode(
            "f_triangle",
            [
                tree_form("d_" + angle2[0]),
                tree_form("d_" + angle2[1]),
                tree_form("d_" + angle2[2]),
            ],
        )
        eq = TreeNode("f_congruent", [angle_1, angle_2])
        eq = str_form(eq)

        for angle in [angle1 + angle2, angle2 + angle1]:
            if sss_rule(*angle) or sas_rule(*angle) or aas_rule(*angle) or rhs_rule(*angle):
                eq_list.append(eq)
                do_cpct()
                return eq
        return None


    def add_angle_equality(h1, h2):
        h1 = plotter_angle_3(h1)
        h2 = plotter_angle_3(h2)
        if h1 == h2:
            return
        row = [Fraction(0)] * len(all_angles)
        row[all_angles.index(h1)] = Fraction(1)
        row[all_angles.index(h2)] = Fraction(-1)
        matrix.append(row)
        matrix_eq.append(Fraction(0))


    def add_line_equality(h1, h2):
        if line_sort(h1) == line_sort(h2):
            return
        row = [Fraction(0)] * (2 + len(line_counter))

        row[index_line(*h1)] = Fraction(1)
        row[index_line(*h2)] = Fraction(-1)
        line_matrix.append(row)
        fix_line_matrix()


    def proof_fx_2(a, b):
        nonlocal eq_list
        nonlocal matrix
        nonlocal matrix_eq
        u, v = a, b
        for item in itertools.combinations(point_pairs, 2):
            if len(set([item[0][0], item[0][1], item[1][0], item[1][1]])) == 4:
                for item2 in itertools.product(item[0], item[1]):
                    if (
                        line_sort(n2a(item2[0]) + n2a(item2[1])) in line_counter_convert()
                        and line_sort(n2a(item2[0]) + n2a(item2[1])) != line_sort(u)
                        and line_sort(n2a(item2[0]) + n2a(item2[1])) != line_sort(v)
                    ):
                        c = None
                        d = None
                        if item[0][0] in item2:
                            c = item[0][1]
                        if item[0][1] in item2:
                            c = item[0][0]
                        if item[1][0] in item2:
                            d = item[1][1]
                        if item[1][1] in item2:
                            d = item[1][0]
                        a, b = item2
                        if (
                            is_same_line(line_sort(n2a(a) + n2a(c)), line_sort(u))
                            and is_same_line(line_sort(n2a(b) + n2a(d)), line_sort(v))
                        ) or (
                            is_same_line(line_sort(n2a(a) + n2a(c)), line_sort(v))
                            and is_same_line(line_sort(n2a(b) + n2a(d)), line_sort(u))
                        ):
                            tmp = find_intersection_3(
                                points[c][0],
                                points[c][1],
                                points[d][0],
                                points[d][1],
                                points[a][0],
                                points[a][1],
                                points[b][0],
                                points[b][1],
                            )
                            if tmp[1] == "intersect":
                                add_angle_equality(
                                    n2a(c) + n2a(a) + n2a(b), n2a(d) + n2a(b) + n2a(a)
                                )


    def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        try:
            if x1 == x2:
                if x3 == x4:
                    return None, "parallel vertical lines"
                m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None
                x = x1
                y = m2 * x + (y3 - m2 * x3) if m2 is not None else None
                return (x, y), "intersect" if y is not None else "no intersection"

            if x3 == x4:
                m1 = (y2 - y1) / (x2 - x1)
                x = x3
                y = m1 * x + (y1 - m1 * x1)
                return (x, y), "intersect"

            m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else None
            m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None

            if m1 == m2:
                return None, "parallel lines"

            if m1 is None:
                x = x1
                y = m2 * x + (y3 - m2 * x3)
            elif m2 is None:
                x = x3
                y = m1 * x + (y1 - m1 * x1)
            else:

                a = m1
                b = y1 - m1 * x1
                c = m2
                d = y3 - m2 * x3
                x = (d - b) / (a - c)
                y = a * x + b

            return (x, y), "intersect"
        except:
            return None, "error"


    def find_intersection_3(x1, y1, x2, y2, x3, y3, x4, y4):

        if x2 == x1 and x4 == x3:
            return None, "error"
        elif x2 == x1:
            x = x1
            m2 = (y4 - y3) / (x4 - x3)
            d = y3 - m2 * x3
            y = m2 * x + d
        elif x4 == x3:
            x = x3
            m1 = (y2 - y1) / (x2 - x1)
            b = y1 - m1 * x1
            y = m1 * x + b
        else:

            m1 = (y2 - y1) / (x2 - x1)
            m2 = (y4 - y3) / (x4 - x3)

            if m1 == m2:
                return None, "error"

            a = m1
            b = y1 - m1 * x1
            c = m2
            d = y3 - m2 * x3
            x = (d - b) / (a - c)
            y = a * x + b

        def is_within(x1, x2, x):
            return min(x1, x2) <= x <= max(x1, x2)

        if (
            is_within(x1, x2, x)
            and is_within(y1, y2, y)
            and is_within(x3, x4, x)
            and is_within(y3, y4, y)
        ):
            return (x, y), "intersect"
        return None, "error"


    def find_intersections_2(points, point_pairs):

        intersections = []
        for item in itertools.combinations(point_pairs, 2):
            x1, y1 = points[item[0][0]]
            x2, y2 = points[item[0][1]]
            x3, y3 = points[item[1][0]]
            x4, y4 = points[item[1][1]]
            tmp = find_intersection_3(x1, y1, x2, y2, x3, y3, x4, y4)
            if tmp[1] == "intersect":
                intersections.append(tmp[0])

        filtered_intersections = [point for point in intersections if point not in points]

        return filtered_intersections


    def a2n(letter):
        return ord(letter) - ord("A")


    def a2n2(line):
        return (a2n(line[0]), a2n(line[1]))


    def find_intersection_line_with_segment(x1, y1, x2, y2, x3, y3, x4, y4):
        def is_within(x1, x2, x):
            return min(x1, x2) <= x <= max(x1, x2)

        ans = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
        output = False
        if (
            ans[1] == "intersect"
            and is_within(x3, x4, ans[0][0])
            and is_within(y3, y4, ans[0][1])
        ):
            output = True
        return output


    def polygon_area(points):
        n = len(points)
        area = Fraction(0)
        for i in range(n - 1):
            area += points[i][0] * points[i + 1][1] - points[i][1] * points[i + 1][0]
        area += points[-1][0] * points[0][1] - points[-1][1] * points[0][0]
        return abs(area) / 2


    def surrounding_angle(given_point):
        def is_enclosed_angle(curr, h1, h2, h3):
            return find_intersection_line_with_segment(
                curr[0], curr[1], h2[0], h2[1], h1[0], h1[1], h3[0], h3[1]
            )

        lst = []
        for line in point_pairs:
            if given_point == line[0]:
                lst.append(points[line[1]])
            elif given_point == line[1]:
                lst.append(points[line[0]])

        for item in itertools.permutations(lst):
            if all(
                is_enclosed_angle(points[given_point], item[i], item[i + 1], item[i + 2])
                for i in range(0, len(item) - 2, 1)
            ):
                lst = list(item)
                break

        tmp = [points.index(x) for x in lst]

        return tmp


    def n2a(number):
        return chr(number + ord("A"))


    def straight_line_2(point_list):
        nonlocal lines
        nonlocal points
        nonlocal point_pairs
        point_list = [a2n(x) for x in point_list]
        tmp = polygon_area([points[x] for x in point_list])
        return tmp == Fraction(0)


    def straight_line(point_list):
        nonlocal lines
        nonlocal points
        nonlocal point_pairs
        tmp = polygon_area([points[x] for x in point_list])

        return tmp == Fraction(0)


    def draw_points_and_lines(
        points,
        lines,
        image_size=(1500, 1500),
        point_radius=5,
        point_color=(0, 0, 0),
        line_color=(255, 0, 0),
        text_color=(0, 0, 0),
    ):

        image = Image.new("RGB", image_size, color="white")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except IOError:
            font = ImageFont.load_default()

        for index, (x, y) in enumerate(points):
            draw.ellipse(
                (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                fill=point_color,
            )

            draw.text(
                (x + point_radius + 5, y - point_radius - 5),
                n2a(index),
                fill=text_color,
                font=font,
            )

        for (x1, y1), (x2, y2) in lines:
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=2)

        return image


    def plotter_text(text, color="black", force=False, auto_next_line=True):
        nonlocal plotter_on
        nonlocal log
        if auto_next_line:
            log += text + "\n"
        else:
            log += text
        return
        if not force and not plotter_on:
            return

        if auto_next_line:
            console.insert(tk.END, text + "\n", color)
        else:
            console.insert(tk.END, text, color)
        console.see(tk.END)


    def display_image(image_path):

        img = Image.open(image_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk


    def plotter_diagram():
        nonlocal lines
        nonlocal points
        nonlocal point_pairs
        return
        image = draw_points_and_lines(points, lines)

        image.save("points_and_lines_image.png")

        display_image("points_and_lines_image.png")


    def travel_till_end(start, step):
        nonlocal lines
        nonlocal points
        nonlocal point_pairs
        done = False
        step_taken = [step]
        while not done:
            done = True
            for item in surrounding_angle(step):
                if (
                    straight_line([step, start, item])
                    and item not in step_taken
                    and start != item
                    and step != start
                    and step != item
                ):
                    step_taken.append(item)
                    step = item

                    done = False
                    break
        return step


    def sur(angle):
        nonlocal lines
        nonlocal points
        nonlocal point_pairs
        count = 0
        if a2n(angle[0]) in surrounding_angle(a2n(angle[1])):
            count += 1
        if a2n(angle[2]) in surrounding_angle(a2n(angle[1])):
            count += 1
        return count


    def plotter_angle(a, b, c, a_do=True, c_do=True):
        nonlocal lines
        nonlocal points
        nonlocal point_pairs

        if a_do:
            a = travel_till_end(b, a)
        else:
            a = travel_till_end(b, a)
            a = travel_till_end(b, a)
        if c_do:
            c = travel_till_end(b, c)
        else:
            c = travel_till_end(b, c)
            c = travel_till_end(b, c)

        m, n = sorted([a, c])
        return n2a(m) + n2a(b) + n2a(n)


    def plotter_angle_2(angle, a_do=True, c_do=True):
        nonlocal lines
        nonlocal points
        nonlocal point_pairs
        x = angle
        return plotter_angle(a2n(x[0]), a2n(x[1]), a2n(x[2]), a_do, c_do)


    def plotter_angle_3(angle):
        lst = [
            plotter_angle_2(angle, True, True),
            plotter_angle_2(angle, True, False),
            plotter_angle_2(angle, False, True),
            plotter_angle_2(angle, False, False),
        ]
        return sorted(lst, key=lambda x: sur(x))[0]


    def plotter_angle_4(a, b, c):
        return plotter_angle_3(n2a(a) + n2a(b) + n2a(c))


    def combine(a, b):
        nonlocal lines
        nonlocal points
        nonlocal point_pairs

        a = plotter_angle_3(a)
        b = plotter_angle_3(b)
        if a[1] != b[1]:
            return None
        if len(set(a + b)) != 4:
            return None
        r = a[0] + a[2] + b[0] + b[2]
        r = r.replace([x for x in r if r.count(x) == 2][0], "")
        out = plotter_angle_3(r[0] + b[1] + r[1])

        return out


    def angle_sort(angle):
        if a2n(angle[0]) > a2n(angle[2]):
            angle = angle[2] + angle[1] + angle[0]
        return angle


    def line_sort(line):
        if a2n(line[0]) > a2n(line[1]):
            line = line[1] + line[0]
        return line


    all_angles = []


    def break_equation(equation):
        sub_equation_list = [equation]
        equation = equation
        for child in equation.children:
            sub_equation_list += break_equation(child)
        return sub_equation_list


    def row_swap(matrix, i, j):

        matrix[i], matrix[j] = matrix[j], matrix[i]


    def row_scale(matrix, i, scale_factor):

        matrix[i] = [scale_factor * elem for elem in matrix[i]]


    def row_addition(matrix, src_row, dest_row, scale_factor):

        matrix[dest_row] = [
            elem_dest + scale_factor * elem_src
            for elem_dest, elem_src in zip(matrix[dest_row], matrix[src_row])
        ]


    def remove_zero_rows(matrix):

        return [row for row in matrix if any(element != Fraction(0) for element in row)]


    def gauss_jordan_elimination(matrix):

        rows = len(matrix)
        cols = len(matrix[0])
        row = 0

        for col in range(cols):

            pivot_row = max(range(row, rows), key=lambda r: abs(matrix[r][col]))

            if matrix[pivot_row][col] == 0:
                continue

            if pivot_row != row:
                row_swap(matrix, row, pivot_row)

            row_scale(matrix, row, Fraction(1, matrix[row][col]))

            for r in range(rows):
                if r != row:
                    row_addition(matrix, row, r, -matrix[r][col])

            row += 1
            if row >= rows:
                break

        return remove_zero_rows(matrix)


    def matrix_to_list(matrix):
        return [list(item) for item in matrix]


    def fix_matrix():
        nonlocal matrix
        for i in range(len(matrix)):
            if len(matrix[i]) < len(all_angles):
                matrix[i] += [Fraction(0)] * (len(all_angles) - len(matrix[i]))


    def fix_line_matrix():
        nonlocal line_matrix
        nonlocal line_counter
        nonlocal point_pairs

        target_size = len(line_counter)

        for i in range(len(line_matrix)):

            if len(line_matrix[i]) < target_size:
                line_matrix[i] += [Fraction(0)] * (target_size - len(line_matrix[i]))

            elif len(line_matrix[i]) > target_size:
                line_matrix[i] = line_matrix[i][:target_size]


    def try_line_matrix():
        nonlocal line_matrix
        if line_matrix == []:
            return

        A = np.array(line_matrix, dtype=object)
        new_matrix = gauss_jordan_elimination(A)

        for item in itertools.combinations(line_counter_convert(), 2):
            row = [Fraction(0)] * (len(line_counter))
            row[index_line(*item[0])] = Fraction(1)
            row[index_line(*item[1])] = Fraction(-1)
            old_matrix = copy.deepcopy(line_matrix)
            line_matrix.append(row)

            if matrices_equal(
                gauss_jordan_elimination(np.array(line_matrix, dtype=object)), new_matrix
            ):

                pass
            else:
                line_matrix = copy.deepcopy(old_matrix)
        fix_line_matrix()


    def matrices_equal(matrix1, matrix2):

        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            return False

        for row1, row2 in zip(matrix1, matrix2):
            if any(x != y for x, y in zip(row1, row2)):
                return False

        return True


    def try_matrix():
        nonlocal matrix
        nonlocal matrix_eq
        nonlocal eq_list
        nonlocal all_angles
        if matrix == []:
            return

        A = np.array(matrix, dtype=object)
        B = np.array(matrix_eq, dtype=object)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        augmented_matrix = np.hstack((A, B))

        new_matrix = gauss_jordan_elimination(augmented_matrix)

        for item in itertools.combinations(all_angles, 2):
            row = [Fraction(0)] * len(matrix[0])
            row[all_angles.index(item[0])] = Fraction(1)
            row[all_angles.index(item[1])] = Fraction(-1)
            matrix.append(row)
            matrix_eq.append(Fraction(0))

            if matrices_equal(
                gauss_jordan_elimination(
                    np.hstack(
                        (
                            np.array(matrix, dtype=object),
                            np.array(matrix_eq, dtype=object).reshape(-1, 1),
                        )
                    )
                ),
                new_matrix,
            ):

                pass
            else:
                matrix.pop(-1)
                matrix_eq.pop(-1)
        for item in matrix_to_list(new_matrix):

            non_zero_count = sum(1 for x in item[:-1] if x != Fraction(0))
            if (
                non_zero_count == 1
                and item.count(Fraction(1)) == 1
                and len(item) == len(item[:-1]) + 1
            ):

                matrix.append(item[:-1])
                matrix_eq.append(item[-1])


    def line_matrix_plotter(plotter_it=True):
        def remove_duplicate_rows(matrix):

            unique_rows = set(tuple(row) for row in matrix)

            return [list(row) for row in unique_rows]

        nonlocal line_matrix
        nonlocal point_pairs

        fix_line_matrix()

        line_matrix = remove_duplicate_rows(line_matrix)

        string = "$"
        for i in range(len(line_matrix)):
            for j in range(len(line_matrix[i])):
                if line_matrix[i][j] != Fraction(0):
                    if line_matrix[i][j] < Fraction(0):
                        string += "-line(" + index_line_matrix(j) + ")"
                    else:
                        string += "+line(" + index_line_matrix(j) + ")"
            string += "=0\n"
        string = string.replace("\n+", "\n").replace("$+", "").replace("$", "")
        if plotter_it and string != "":
            plotter_text(string, "black", False, False)


    def matrix_plotter(plotter_it=True):
        def remove_duplicates(matrix_2d, array_1d):
            unique_rows = {}
            for row, val in zip(matrix_2d, array_1d):
                row_tuple = tuple(row)
                if row_tuple not in unique_rows:
                    unique_rows[row_tuple] = val
            new_matrix_2d = list(unique_rows.keys())
            new_array_1d = list(unique_rows.values())
            return new_matrix_2d, new_array_1d

        nonlocal matrix
        nonlocal matrix_eq
        nonlocal all_angles
        fix_matrix()
        matrix, matrix_eq = remove_duplicates(matrix, matrix_eq)
        for i in range(len(matrix)):
            matrix[i] = list(matrix[i])

        string = "$"
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] != Fraction(0):
                    if matrix[i][j] == Fraction(-1):
                        string += "-angle(" + all_angles[j] + ")"
                    elif matrix[i][j] == Fraction(1):
                        string += "+angle(" + all_angles[j] + ")"
                    else:
                        string += (
                            "+" + str(float(matrix[i][j])) + "*angle(" + all_angles[j] + ")"
                        )
            string += "=" + str(int(matrix_eq[i])) + "\n"
        string = string.replace("\n+", "\n").replace("$+", "").replace("$", "")
        if plotter_it and string != "":
            plotter_text(string, "black", False, False)


    def process(plotter_it=True):
        nonlocal lines
        nonlocal points
        nonlocal point_pairs
        nonlocal all_angles
        nonlocal matrix
        nonlocal matrix_eq
        nonlocal line_matrix
        nonlocal eq_list

        lines = [(points[start], points[end]) for start, end in point_pairs]
        plotter_diagram()
        output = []
        find = find_intersections_2(points, point_pairs)

        for i in range(len(points)):

            for angle in itertools.combinations(surrounding_angle(i), 2):

                if angle[0] != angle[1]:

                    output.append(plotter_angle_4(angle[0], i, angle[1]))

        output = list(set(output))
        append_angles = set(output) - set(all_angles)
        reject_angles = set(all_angles) - set(output)

        all_angles = all_angles + list(append_angles)

        index_iter = sorted([all_angles.index(item) for item in reject_angles])[::-1]

        for i in range(len(matrix) - 1, -1, -1):
            for item in index_iter:
                if int(matrix[i][item]) != Fraction(0):
                    matrix.pop(i)
                    matrix_eq.pop(i)
                    break
        for item in index_iter:
            for i in range(len(matrix)):
                matrix[i].pop(item)
            all_angles.pop(item)
        for item in point_pairs:
            s = n2a(item[0]) + n2a(item[1])
            index_line(*s)
        output = []

        for i in range(len(points)):
            for angle in all_angles:
                if straight_line([a2n(x) for x in angle]):

                    output.append(angle)
        output = list(set(output))
        for x in output:

            index_line(*line_sort(x[0] + x[1]))
            matrix.append([Fraction(0)] * len(all_angles))
            matrix[-1][all_angles.index(x)] = Fraction(1)
            matrix_eq.append(Fraction(180))

            row = [Fraction(0)] * (len(line_counter) + 3)
            row[index_line(x[0], x[1])] = Fraction(1)
            row[index_line(x[1], x[2])] = Fraction(1)
            row[index_line(x[2], x[0])] = Fraction(-1)
            line_matrix.append(row)

        for angle in itertools.permutations(all_angles, 3):
            if combine(angle[0], angle[1]) == angle[2]:
                go_to_next = False
                if (
                    straight_line_2(angle[0])
                    or straight_line_2(angle[1])
                    or straight_line_2(angle[2])
                ):
                    if straight_line_2(angle[2]):
                        go_to_next = True
                    else:
                        continue
                if not go_to_next:

                    hhh = [
                        a2n(h)
                        for h in list(set(angle[0] + angle[1] + angle[2]))
                        if list(angle[0] + angle[1] + angle[2]).count(h) == 3
                    ][0]
                    hh = [
                        (a2n(h), hhh)
                        for h in list(set(angle[0] + angle[1] + angle[2]))
                        if list(angle[0] + angle[1] + angle[2]).count(h) == 2
                    ]
                    orig = copy.deepcopy(point_pairs)
                    point_pairs = hh
                    hh = surrounding_angle(hhh)
                    point_pairs = copy.deepcopy(orig)

                if go_to_next or n2a(hh[1]) not in angle[2]:
                    matrix.append([Fraction(0)] * len(all_angles))
                    matrix[-1][all_angles.index(angle[0])] = Fraction(1)
                    matrix[-1][all_angles.index(angle[1])] = Fraction(1)
                    matrix[-1][all_angles.index(angle[2])] = Fraction(-1)
                    matrix_eq.append(Fraction(0))

        for angle in itertools.combinations(all_angles, 2):
            if (
                angle[0][1] == angle[1][1]
                and straight_line([a2n(x) for x in angle[0]])
                and straight_line([a2n(x) for x in angle[1]])
            ):
                tmp1 = plotter_angle_3(angle[1][0] + angle[0][1] + angle[0][2])
                tmp2 = plotter_angle_3(angle[0][0] + angle[1][1] + angle[1][2])
                matrix.append([Fraction(0)] * len(all_angles))
                matrix[-1][all_angles.index(tmp1)] = Fraction(1)
                matrix[-1][all_angles.index(tmp2)] = Fraction(-1)
                matrix_eq.append(Fraction(0))

                tmp1 = plotter_angle_3(angle[1][2] + angle[0][1] + angle[0][2])
                tmp2 = plotter_angle_3(angle[1][0] + angle[1][1] + angle[0][0])
                matrix.append([Fraction(0)] * len(all_angles))
                matrix[-1][all_angles.index(tmp1)] = Fraction(1)
                matrix[-1][all_angles.index(tmp2)] = Fraction(-1)
                matrix_eq.append(Fraction(0))

        generate_all_lines()
        fix_line_matrix()

        all_triangle(all_angles)
        do_isoceles()
        matrix_plotter(plotter_it)
        line_matrix_plotter(plotter_it)
        eq_list = list(set(eq_list))
        if plotter_it:
            for item in eq_list:
                plotter_text(string_equation(item))


    def all_cycle(graph):
        nonlocal lines, matrix, matrix_eq, points, point_pairs, all_angles
        cycles = []

        def findNewCycles(path):
            start_node = path[0]
            next_node = None
            for edge in graph:
                node1, node2 = edge
                if start_node in edge:
                    next_node = node2 if node1 == start_node else node1
                    if not visited(next_node, path):
                        sub_path = [next_node] + path
                        findNewCycles(sub_path)
                    elif len(path) > 2 and next_node == path[-1]:
                        p = rotate_to_smallest(path)
                        inv = invert(p)
                        if isNew(p) and isNew(inv):
                            cycles.append(p)

        def invert(path):
            return rotate_to_smallest(path[::-1])

        def rotate_to_smallest(path):
            n = path.index(min(path))
            return path[n:] + path[:n]

        def isNew(path):
            return path not in cycles

        def visited(node, path):
            return node in path

        for edge in graph:
            for node in edge:
                findNewCycles([node])
        return cycles


    def perpendicular_line_intersection(segment_start, segment_end, point):
        x1, y1 = segment_start
        x2, y2 = segment_end
        xp, yp = point

        if x2 == x1:
            xq = x1
            yq = yp

        elif y2 == y1:
            xq = xp
            yq = y1

        else:

            m = (y2 - y1) / (x2 - x1)

            m_perp = -1 / m

            xq = (m * x1 - m_perp * xp + yp - y1) / (m - m_perp)

            yq = m * (xq - x1) + y1

        return (xq, yq)


    def is_reflex_vertex(polygon, vertex_index):
        prev_index = (vertex_index - 1) % len(polygon)
        next_index = (vertex_index + 1) % len(polygon)
        modified_polygon = polygon[:vertex_index] + polygon[vertex_index + 1 :]
        original_area = polygon_area(polygon)
        modified_area = polygon_area(modified_polygon)
        if modified_area <= original_area:
            return False
        else:
            return True


    def is_reflex_by_circle(polygon):
        output = []
        for i in range(len(polygon)):
            if is_reflex_vertex(polygon, i):
                output.append(i)
        return output


    def all_triangle(all_angles):
        nonlocal lines
        nonlocal matrix
        nonlocal matrix_eq
        nonlocal points
        nonlocal point_pairs
        nonlocal all_tri

        cycle = all_cycle(point_pairs)

        new_cycle = []
        for item in cycle:
            remove_item = []
            for i in range(-2, len(item) - 2, 1):
                if straight_line([item[i], item[i + 1], item[i + 2]]):
                    remove_item.append(item[i + 1])
            new_item = item
            for i in range(len(new_item) - 1, -1, -1):
                if new_item[i] in remove_item:
                    new_item.pop(i)
            new_cycle.append(new_item)
        for x in new_cycle:

            to_remove = []
            for i in range(-2, len(x) - 2, 1):
                angle = [x[i], x[i + 1], x[i + 2]]
                if straight_line(angle):
                    to_remove.append(i)
            to_remove = sorted(to_remove)[::-1]

            for item in to_remove:
                x.pop(item)

            convex_angle = is_reflex_by_circle([points[y] for y in x])

            out = []
            v = None
            for i in range(-2, len(x) - 2, 1):
                angle = [x[i], x[i + 1], x[i + 2]]
                tmp = [[z for z in x][y] for y in convex_angle]

                v = "".join([n2a(y) for y in angle])
                if angle[1] in tmp:
                    out.append(
                        "(360-" + plotter_angle_3("".join([n2a(y) for y in angle])) + ")"
                    )
                else:
                    out.append(plotter_angle_3("".join([n2a(y) for y in angle])))

            if len(x) == 3:
                all_tri.append(v)
            if out == []:
                continue

            copy_out = copy.deepcopy(out)

            out = copy.deepcopy(copy_out)
            for i in range(len(out)):
                out[i] = out[i].replace("(360-", "").replace(")", "")

            matrix.append([Fraction(0)] * len(all_angles))
            subtract = 0

            for i in range(len(out)):
                if "(360-" in copy_out[i]:

                    subtract += 360
                    matrix[-1][all_angles.index(out[i])] = Fraction(-1)
                else:
                    matrix[-1][all_angles.index(out[i])] = Fraction(1)
            matrix_eq.append(Fraction(180 * (len(x) - 2) - subtract))

        all_tri = list(set(all_tri))


    def fix_angle_line(eq):
        nonlocal all_angles

        eq = tree_form(eq)
        content = list(set("".join(all_angles)))
        for item2 in itertools.permutations(content, 3):
            item2 = "".join(item2)
            if item2 not in string_equation(str_form(eq)):
                continue

            eq = replace(eq, convert_angle_2(item2), convert_angle_2(plotter_angle_3(item2)))
            eq = replace(
                eq,
                tree_form(str_form(convert_angle_2(item2)).replace("f_angle", "f_xangle")),
                tree_form(
                    str_form(convert_angle_2(plotter_angle_3(item2))).replace(
                        "f_angle", "f_xangle"
                    )
                ),
            )
        for item in point_pairs:
            item = n2a(item[0]) + n2a(item[1])
            eq = replace(eq, line_fx(item), line_fx(line_sort(item)))
            eq = replace(eq, line_fx(item[1] + item[0]), line_fx(line_sort(item)))
            eq = replace(
                eq,
                tree_form(str_form(line_fx(item)).replace("f_line", "f_xline")),
                tree_form(str_form(line_fx(line_sort(item))).replace("f_line", "f_xline")),
            )
            eq = replace(
                eq,
                tree_form(
                    str_form(line_fx(item[1] + item[0])).replace("f_line", "f_xline")
                ),
                tree_form(str_form(line_fx(line_sort(item))).replace("f_line", "f_xline")),
            )
        return str_form(eq)


    def extend(line, point_start, distance):
        nonlocal points
        b = None
        a = points[a2n(point_start)]
        if line[0] == point_start:
            b = points[a2n(line[1])]
        else:
            b = points[a2n(line[0])]
        ba = [a[0] - b[0], a[1] - b[1]]
        length_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        unit_vector_ba = [ba[0] / length_ba, ba[1] / length_ba]
        bc = [unit_vector_ba[0] * distance, unit_vector_ba[1] * distance]
        c = tuple([Fraction(round(a[0] + bc[0])), Fraction(round(a[1] + bc[1]))])
        out = c
        if polygon_area([a, b, c]) != Fraction(0):
            out = perpendicular_line_intersection(a, b, c)
        points.append(out)
        plotter_text("new point added")


    def divide_line(line, new_val=None):
        nonlocal lines
        nonlocal matrix
        nonlocal matrix_eq
        nonlocal points
        nonlocal point_pairs
        a = a2n(line[0])
        b = a2n(line[1])
        if (a, b) not in point_pairs:
            a, b = b, a
            if (a, b) not in point_pairs:
                return None
        new_point = None
        if new_val is None:
            new_point = (
                round((points[a][0] + points[b][0]) / 2),
                round((points[a][1] + points[b][1]) / 2),
            )
        else:
            new_point = new_val

        point_pairs.pop(point_pairs.index((a, b)))
        point_pairs.append((len(points), a))
        point_pairs.append((len(points), b))
        points.append((Fraction(new_point[0]), Fraction(new_point[1])))


    def is_point_on_line(line, point):

        a = points[line[0]]
        b = points[line[1]]
        c = point

        return polygon_area([a, b, c]) == Fraction(0)


    def find_line_for_point(point):

        nonlocal point_pairs
        output = []
        for i, line in enumerate(point_pairs):
            if is_point_on_line(line, point):
                output.append(i)
        return output


    def connect_point(point_ab):
        nonlocal lines
        nonlocal points
        nonlocal point_pairs
        nonlocal eq_list
        output = []
        point_a, point_b = point_ab
        point_pairs.append((a2n(point_a), a2n(point_b)))

        inter = find_intersections_2(points, point_pairs)

        for p in inter:

            item_list = find_line_for_point(p)

            points.append(p)
            to_remove = []
            to_add = []
            for item in item_list:
                a, b = point_pairs[item]
                to_remove.append(point_pairs.index((a, b)))
                to_add.append((len(points) - 1, a))
                to_add.append((len(points) - 1, b))
            a1, a2, a3, a4 = to_add[0][1], to_add[1][1], to_add[2][1], to_add[3][1]
            s1 = plotter_angle_4(a1, len(points) - 1, a4), plotter_angle_4(
                a3, len(points) - 1, a2
            )
            s2 = plotter_angle_4(a1, len(points) - 1, a3), plotter_angle_4(
                a4, len(points) - 1, a2
            )

            output.append(s1)

            output.append(s2)

            to_remove = sorted(to_remove)[::-1]
            for item in to_remove:
                point_pairs.pop(item)
            for item in to_add:
                point_pairs.append(item)

        return output


    def draw_triangle():
        nonlocal points
        nonlocal point_pairs
        points = [
            (Fraction(400), Fraction(800)),
            (Fraction(800), Fraction(750)),
            (Fraction(600), Fraction(400)),
        ]

        point_pairs = [(0, 1), (1, 2), (2, 0)]


    def perpendicular(point, line, ver=1):
        nonlocal points
        nonlocal point_pairs
        nonlocal eq_list
        nonlocal all_angles
        output = None

        if ver == 1:
            output = perpendicular_line_intersection(
                points[a2n(line[0])], points[a2n(line[1])], points[a2n(point)]
            )
        else:
            output = perpendicular_line_intersection2(
                points[a2n(line[0])], points[a2n(line[1])], points[a2n(point)]
            )
        divide_line(line, output)
        num = len(points) - 1

        tmp = connect_point(n2a(len(points) - 1) + point)
        process(False)
        for item in tmp:
            row = [Fraction(0)] * len(all_angles)
            row[all_angles.index(item[0])] = Fraction(1)
            row[all_angles.index(item[1])] = Fraction(-1)
            matrix.append(row)
            matrix_eq.append(Fraction(0))

        eq1 = plotter_angle_3(point + n2a(num) + line[0])
        eq2 = plotter_angle_3(point + n2a(num) + line[1])
        row = [Fraction(0)] * len(all_angles)
        row[all_angles.index(eq1)] = Fraction(1)
        matrix_eq.append(Fraction(90))
        matrix.append(row)
        row = [Fraction(0)] * len(all_angles)
        row[all_angles.index(eq2)] = Fraction(1)
        matrix_eq.append(Fraction(90))
        matrix.append(row)


    def load_image(file_path):
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk


    def convert_form(eq):
        def num_not_var(eq):
            for index, item in enumerate(all_angles):
                eq = replace(eq, convert_angle(item), tree_form("v_" + str(index)))
            return eq

        if all(
            x not in eq
            for x in [
                "f_triangle",
                "f_xtriangle",
                "f_xangle",
                "f_congruent",
                "f_xcongruent",
                "f_line",
                "f_xline",
            ]
        ):
            return str_form(num_not_var(tree_form(eq)))
        return None


    def convert_form_2(eq):
        nonlocal line_counter

        def num_not_var(eq):
            for index, item in enumerate(line_counter):
                eq = replace(
                    eq, line_fx(index_line_matrix(index)), tree_form("v_" + str(index))
                )
            return eq

        if all(
            x not in eq
            for x in [
                "f_triangle",
                "f_xtriangle",
                "f_xangle",
                "f_angle",
                "f_congruent",
                "f_xcongruent",
                "f_xline",
            ]
        ):
            return str_form(num_not_var(tree_form(eq)))
        return None


    def find_all_paths(graph, start_node, end_node, path=[]):
        path = path + [start_node]
        if start_node == end_node:
            return [path]
        if start_node not in graph:
            return []
        paths = []
        for neighbor in graph[start_node]:
            if neighbor not in path:
                new_paths = find_all_paths(graph, neighbor, end_node, path)
                for p in new_paths:
                    paths.append(p)
        return paths


    def generate_graph():
        graph = dict()
        for i in range(len(points)):
            graph[n2a(i)] = [n2a(x) for x in surrounding_angle(i)]
        return graph


    def generate_all_lines():
        for item in itertools.combinations(range(len(points)), 2):
            for path in find_all_paths(generate_graph(), item[0], item[1]):
                if straight_line_2(path):
                    for item2 in itertools.combinations(path, 2):
                        index_line(item2[0], item2[1])


    def is_same_line(line1, line2):
        if line1 == line2:
            return True
        for item in itertools.combinations(range(len(points)), 2):
            for path in find_all_paths(generate_graph(), item[0], item[1]):
                if (
                    straight_line_2(path)
                    and line1[0] in path
                    and line1[1] in path
                    and line2[0] in path
                    and line2[1] in path
                ):
                    return True
        return False


    def do_isoceles():
        nonlocal line_matrix
        nonlocal all_angles
        nonlocal matrix
        nonlocal matrix_eq
        nonlocal all_tri
        lst = line_counter_convert()
        for item in line_matrix:
            if (
                item.count(Fraction(0)) == len(item) - 2
                and item.count(Fraction(1)) == 1
                and item.count(Fraction(-1)) == 1
            ):
                line1 = lst[item.index(1)]
                line2 = lst[item.index(-1)]
                for tri in all_tri:
                    if (
                        line1[0] in tri
                        and line2[0] in tri
                        and line1[1] in tri
                        and line2[1] in tri
                    ):
                        common = set(line1) & set(line2)
                        common = list(common)[0]
                        a = list(set(line1) - set(common))[0]
                        b = list(set(line2) - set(common))[0]
                        row = [Fraction(0)] * len(all_angles)
                        row[all_angles.index(plotter_angle_3(common + a + b))] = Fraction(1)
                        row[all_angles.index(plotter_angle_3(common + b + a))] = Fraction(-1)
                        matrix.append(row)
                        matrix_eq.append(Fraction(0))
                        break


    def do_cpct():
        nonlocal eq_list
        nonlocal matrix
        nonlocal matrix_eq
        cpct_output = []
        for item in eq_list:
            if "congruent" in item:
                m = re.findall(r"[A-Z]{3}", string_equation(item))
                m_list = []
                for item2 in itertools.permutations(range(3)):
                    m_list.append(
                        [
                            m[0][item2[0]] + m[0][item2[1]] + m[0][item2[2]],
                            m[1][item2[0]] + m[1][item2[1]] + m[1][item2[2]],
                        ]
                    )
                for item2 in m_list:
                    add_angle_equality(item2[0], item2[1])
                    add_line_equality(item2[0][:-1], item2[1][:-1])
        fix_matrix()


    def walk_parallel_function(command):
        nonlocal points
        nonlocal point_pairs
        nonlocal eq_list
        nonlocal plotter_on
        nonlocal matrix
        nonlocal matrix_eq
        nonlocal all_tri
        nonlocal line_matrix
        nonlocal line_counter
        nonlocal log
        plotter_on = True
        line_counter = []
        all_tri = []
        points = []
        point_pairs = []
        eq_list = []
        lines = []
        matrix = []
        matrix_eq = []
        line_matrix = []
        log = ""
        command = command.split("\n")
        string = None
        while True:
            if string not in {"show", "hide"}:
                process()
            if command != []:
                string = command.pop(0)
                plotter_text(">>> ", "green", True, False)
                plotter_text(string, "blue", True, True)
            else:
                plotter_text("\nend of program", "green", True, True)
                return log
            if string == "hide":
                plotter_on = False
            if string == "show":
                plotter_on = True
            if string[:13] == "draw triangle":
                draw_triangle()
            elif string == "draw quadrilateral":
                points = [
                    (Fraction(400), Fraction(800)),
                    (Fraction(800), Fraction(750)),
                    (Fraction(600), Fraction(400)),
                    (Fraction(400), Fraction(550)),
                ]
                point_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
            elif string == "compute":
                try_line_matrix()

                def two(angle1, angle2):
                    list1 = list(itertools.permutations(list(angle1)))
                    list2 = list(itertools.permutations(list(angle2)))
                    for i in range(len(list1)):
                        for j in range(len(list2)):
                            out = proof_fx_3("".join(list1[i]), "".join(list2[j]))
                            if out is not None:

                                return

                if len(all_tri) > 1:

                    for item in itertools.combinations(all_tri, 2):

                        two(item[0], item[1])

                output = try_matrix()
            elif string == "draw right triangle":
                points = [(100, 400), (300, 400), (300, 200)]
                point_pairs = [(0, 1), (1, 2), (2, 0)]
            elif string == "draw quadrilateral":
                points = [(272, 47), (8, 211), (289, 380), (422, 62)]
                point_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
            elif string.split(" ")[0] == "perpendicular" and string.split(" ")[2] == "to":
                perpendicular(string.split(" ")[1], string.split(" ")[3])
            elif string == "calculate":
                output = try_matrix()
                eq_list = list(set(eq_list))
            elif string.split(" ")[0] == "prove":
                proof = parser_4.take_input(string.split(" ")[1])
                output = proof_fx(proof)
                if output:
                    eq_list.append(output)
            elif (
                string.split(" ")[0] == "extend"
                and string.split(" ")[2] == "from"
                and string.split(" ")[4] == "for"
            ):
                extend(
                    string.split(" ")[1], string.split(" ")[3], int(string.split(" ")[5])
                )
            elif string.split(" ")[0] == "extend" and string.split(" ")[2] == "to":
                val = find_intersection(
                    points[a2n(string.split(" ")[1][0])][0],
                    points[a2n(string.split(" ")[1][0])][1],
                    points[a2n(string.split(" ")[1][1])][0],
                    points[a2n(string.split(" ")[1][1])][1],
                    points[a2n(string.split(" ")[3][0])][0],
                    points[a2n(string.split(" ")[3][0])][1],
                    points[a2n(string.split(" ")[3][1])][0],
                    points[a2n(string.split(" ")[3][1])][1],
                )
                divide_line(string.split(" ")[3], val[0])

            elif string.split(" ")[0] == "split":
                divide_line(string.split(" ")[-1])
            elif string == "cpct":
                do_cpct()
            elif string.split(" ")[0] == "join":
                list_join = string.split(" ")[1:]
                tmp = None
                for join_iter in list_join:
                    tmp = connect_point(join_iter)
                process(False)
                for item in tmp:
                    row = [Fraction(0)] * len(all_angles)
                    row[all_angles.index(item[0])] = Fraction(1)
                    row[all_angles.index(item[1])] = Fraction(-1)
                    matrix.append(row)
                    matrix_eq.append(Fraction(0))
            elif string.split(" ")[0] == "equation":

                eq_type = string.split(" ")[1]
                if "angle_eq" == eq_type:
                    a = plotter_angle_3(string.split(" ")[2])
                    b = plotter_angle_3(string.split(" ")[3])
                    row = [Fraction(0)] * len(all_angles)
                    row[all_angles.index(a)] = Fraction(1)
                    row[all_angles.index(b)] = Fraction(-1)
                    matrix.append(row)
                    matrix_eq.append(Fraction(0))
                elif "angle_val" == eq_type:
                    a = plotter_angle_3(string.split(" ")[2])
                    val = int(string.split(" ")[3])
                    row = [Fraction(0)] * len(all_angles)
                    row[all_angles.index(a)] = Fraction(1)
                    matrix.append(row)
                    matrix_eq.append(Fraction(val))
                elif "parallel_line" == eq_type:
                    a = line_sort(string.split(" ")[2])
                    b = line_sort(string.split(" ")[3])
                    eq = str_form(
                        TreeNode(
                            "f_parallel", [line_fx(line_sort(a)), line_fx(line_sort(b))]
                        )
                    )
                    eq_list.append(eq)
                    proof_fx_2(a, b)
                elif "line_eq" == eq_type:
                    a = line_sort(string.split(" ")[2])
                    b = line_sort(string.split(" ")[3])
                    row = [Fraction(0)] * (len(all_angles) + 2)
                    row[index_line(*a)] = Fraction(1)
                    row[index_line(*b)] = Fraction(-1)
                    line_matrix.append(row)
                    fix_line_matrix()
    return walk_parallel_function(command)


def ENGINE_R(command):
    plotter_on = True
    line_counter = []
    all_tri = []
    points = []
    point_pairs = []
    eq_list = []
    lines = []
    matrix = []
    matrix_eq = []
    line_matrix = []


    class TreeNode:
        def __init__(self, name, children=None):
            self.name = name
            self.children = children or []


    def tree_form(tabbed_strings):
        lines = tabbed_strings.split("\n")
        root = TreeNode("Root")
        current_level_nodes = {0: root}
        stack = [root]
        for line in lines:
            level = line.count(" ")
            node_name = line.strip()
            node = TreeNode(node_name)
            while len(stack) > level + 1:
                stack.pop()
            parent_node = stack[-1]
            parent_node.children.append(node)
            current_level_nodes[level] = node
            stack.append(node)
        return root.children[0]


    def str_form(node):
        def recursive_str(node, depth=0):
            result = "{}{}".format(" " * depth, node.name)
            for child in node.children:
                result += "\n" + recursive_str(child, depth + 1)
            return result

        return recursive_str(node)


    def line_fx(line_input):
        a = line_input[0]
        b = line_input[1]
        return TreeNode("f_line", [tree_form("d_" + a), tree_form("d_" + b)])


    def index_line(a, b):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles
        a, b = line_sort(a + b)
        a, b = a2n(a), a2n(b)
        if (a, b) in line_counter:
            return line_counter.index((a, b))
        else:
            line_counter.append((a, b))
            fix_line_matrix()
            return len(line_counter) - 1


    def index_line_matrix(index):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles
        a, b = line_counter[index]
        return line_sort(n2a(a) + n2a(b))


    def string_equation_helper(equation_tree):
        if equation_tree.children == []:
            return equation_tree.name
        s = "("
        if len(equation_tree.children) == 1 or equation_tree.name in [
            "f_if",
            "f_xcongruent",
            "f_congruent",
            "f_triangle",
            "f_xangle",
            "f_xline",
            "f_angle",
            "f_line",
            "f_parallel",
        ]:
            s = equation_tree.name[2:] + s
        sign = {
            "f_if": ",",
            "f_xparallel": ",",
            "f_xcongruent": ",",
            "f_congruent": ",",
            "f_triangle": "?",
            "f_add": "+",
            "f_and": "^",
            "f_dif": "?",
            "f_mul": "*",
            "f_eq": "=",
            "f_sub": "-",
            "f_angle": "?",
            "f_xangle": "?",
            "f_parallel": ",",
            "f_xline": "?",
            "f_exist": "?",
            "f_line": "?",
        }
        for child in equation_tree.children:
            s += string_equation_helper(copy.deepcopy(child)) + sign[equation_tree.name]
        s = s[:-1] + ")"
        return s


    def string_equation(eq):
        eq = eq.replace("d_", "")

        return string_equation_helper(tree_form(eq)).replace("?", "")


    def line_eq(line1, line2):
        if line1 == line2:
            return True
        line1 = a2n(line1[0]), a2n(line1[1])
        line2 = a2n(line2[0]), a2n(line2[1])
        row = [Fraction(0)] * len(line_counter)
        row[line_counter.index(line1)] = Fraction(1)
        row[line_counter.index(line2)] = Fraction(-1)
        if row in line_matrix:
            return True
        row[line_counter.index(line1)] = Fraction(-1)
        row[line_counter.index(line2)] = Fraction(1)
        if row in line_matrix:
            return True
        return False


    def angle_eq(angle1, angle2):
        if angle1 == angle2:
            return True

        row = [Fraction(0)] * len(all_angles)
        row[all_angles.index(angle1)] = Fraction(1)
        row[all_angles.index(angle2)] = Fraction(-1)
        if row in matrix and matrix_eq[matrix.index(row)] == Fraction(0):
            return True
        row[all_angles.index(angle1)] = Fraction(-1)
        row[all_angles.index(angle2)] = Fraction(1)
        if row in matrix and matrix_eq[matrix.index(row)] == Fraction(0):
            return True
        return False


    def angle_per(angle):
        row = [Fraction(0)] * len(all_angles)
        row[all_angles.index(angle)] = Fraction(1)
        if row in matrix and matrix_eq[matrix.index(row)] == Fraction(90):
            return True
        return False


    def line_counter_convert():
        output = []
        for item in line_counter:
            output.append(line_sort(n2a(item[0]) + n2a(item[1])))
        return output


    def sss_rule(a1, a2, a3, b1, b2, b3):
        line = [
            line_sort(a1 + a2),
            line_sort(b1 + b2),
            line_sort(a2 + a3),
            line_sort(b2 + b3),
            line_sort(a1 + a3),
            line_sort(b1 + b3),
        ]

        for item in line:
            if item not in line_counter_convert():
                return False

        return (
            line_eq(line[0], line[1])
            and line_eq(line[2], line[3])
            and line_eq(line[4], line[5])
        )


    def sas_rule(a1, a2, a3, b1, b2, b3):
        line = [
            line_sort(a1 + a2),
            line_sort(b1 + b2),
            line_sort(a2 + a3),
            line_sort(b2 + b3),
        ]
        angle = [plotter_angle_3(a1 + a2 + a3), plotter_angle_3(b1 + b2 + b3)]

        for item in line:
            if item not in line_counter_convert():
                return False
        for item in angle:
            if item not in all_angles:

                return False

        return (
            line_eq(line[0], line[1])
            and angle_eq(angle[0], angle[1])
            and line_eq(line[2], line[3])
        )


    def aas_rule(a1, a2, a3, b1, b2, b3):
        line = [line_sort(a2 + a3), line_sort(b2 + b3)]
        angle = [
            plotter_angle_3(a1 + a2 + a3),
            plotter_angle_3(b1 + b2 + b3),
            plotter_angle_3(a3 + a1 + a2),
            plotter_angle_3(b3 + b1 + b2),
        ]

        for item in line:
            if item not in line_counter_convert():
                return False

        for item in angle:
            if item not in all_angles:
                return False

        return (
            line_eq(line[0], line[1])
            and angle_eq(angle[0], angle[1])
            and angle_eq(angle[2], angle[3])
        )


    def rhs_rule(a1, a2, a3, b1, b2, b3):
        line = [
            line_sort(a1 + a2),
            line_sort(b1 + b2),
            line_sort(a1 + a3),
            line_sort(b1 + b3),
        ]
        angle = [plotter_angle_3(a1 + a2 + a3), plotter_angle_3(b1 + b2 + b3)]

        for item in line:
            if item not in line_counter_convert():
                return False

        for item in angle:
            if item not in all_angles:
                return False

        return (
            line_eq(line[0], line[1])
            and angle_eq(angle[0], angle[1])
            and line_eq(line[2], line[3])
            and angle_per(angle[0])
        )


    def proof_fx_3(angle1, angle2):
        nonlocal eq_list

        angle_1 = TreeNode(
            "f_triangle",
            [
                tree_form("d_" + angle1[0]),
                tree_form("d_" + angle1[1]),
                tree_form("d_" + angle1[2]),
            ],
        )
        angle_2 = TreeNode(
            "f_triangle",
            [
                tree_form("d_" + angle2[0]),
                tree_form("d_" + angle2[1]),
                tree_form("d_" + angle2[2]),
            ],
        )
        eq = TreeNode("f_congruent", [angle_1, angle_2])
        eq = str_form(eq)

        for angle in [angle1 + angle2, angle2 + angle1]:
            if sss_rule(*angle) or sas_rule(*angle) or aas_rule(*angle) or rhs_rule(*angle):
                eq_list.append(eq)
                do_cpct()
                return eq
        return None


    def add_angle_equality(h1, h2):
        h1 = plotter_angle_3(h1)
        h2 = plotter_angle_3(h2)
        if h1 == h2:
            return
        row = [Fraction(0)] * len(all_angles)
        row[all_angles.index(h1)] = Fraction(1)
        row[all_angles.index(h2)] = Fraction(-1)
        matrix.append(row)
        matrix_eq.append(Fraction(0))


    def add_line_equality(h1, h2):
        if line_sort(h1) == line_sort(h2):
            return
        row = [Fraction(0)] * (2 + len(line_counter))

        row[index_line(*h1)] = Fraction(1)
        row[index_line(*h2)] = Fraction(-1)
        line_matrix.append(row)
        fix_line_matrix()


    def proof_fx_2(a, b):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles
        u, v = a, b
        for item in itertools.combinations(point_pairs, 2):
            if len(set([item[0][0], item[0][1], item[1][0], item[1][1]])) == 4:
                for item2 in itertools.product(item[0], item[1]):
                    if (
                        line_sort(n2a(item2[0]) + n2a(item2[1])) in line_counter_convert()
                        and line_sort(n2a(item2[0]) + n2a(item2[1])) != line_sort(u)
                        and line_sort(n2a(item2[0]) + n2a(item2[1])) != line_sort(v)
                    ):
                        c = None
                        d = None
                        if item[0][0] in item2:
                            c = item[0][1]
                        if item[0][1] in item2:
                            c = item[0][0]
                        if item[1][0] in item2:
                            d = item[1][1]
                        if item[1][1] in item2:
                            d = item[1][0]
                        a, b = item2
                        if (
                            is_same_line(line_sort(n2a(a) + n2a(c)), line_sort(u))
                            and is_same_line(line_sort(n2a(b) + n2a(d)), line_sort(v))
                        ) or (
                            is_same_line(line_sort(n2a(a) + n2a(c)), line_sort(v))
                            and is_same_line(line_sort(n2a(b) + n2a(d)), line_sort(u))
                        ):
                            tmp = find_intersection_3(
                                points[c][0],
                                points[c][1],
                                points[d][0],
                                points[d][1],
                                points[a][0],
                                points[a][1],
                                points[b][0],
                                points[b][1],
                            )
                            if tmp[1] == "intersect":
                                add_angle_equality(
                                    n2a(c) + n2a(a) + n2a(b), n2a(d) + n2a(b) + n2a(a)
                                )


    def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        try:
            if x1 == x2:
                if x3 == x4:
                    return None, "parallel vertical lines"
                m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None
                x = x1
                y = m2 * x + (y3 - m2 * x3) if m2 is not None else None
                return (x, y), "intersect" if y is not None else "no intersection"

            if x3 == x4:
                m1 = (y2 - y1) / (x2 - x1)
                x = x3
                y = m1 * x + (y1 - m1 * x1)
                return (x, y), "intersect"

            m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else None
            m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None

            if m1 == m2:
                return None, "parallel lines"

            if m1 is None:
                x = x1
                y = m2 * x + (y3 - m2 * x3)
            elif m2 is None:
                x = x3
                y = m1 * x + (y1 - m1 * x1)
            else:

                a = m1
                b = y1 - m1 * x1
                c = m2
                d = y3 - m2 * x3
                x = (d - b) / (a - c)
                y = a * x + b

            return (x, y), "intersect"
        except:
            return None, "error"


    def find_intersection_3(x1, y1, x2, y2, x3, y3, x4, y4):

        if x2 == x1 and x4 == x3:
            return None, "error"
        elif x2 == x1:
            x = x1
            m2 = (y4 - y3) / (x4 - x3)
            d = y3 - m2 * x3
            y = m2 * x + d
        elif x4 == x3:
            x = x3
            m1 = (y2 - y1) / (x2 - x1)
            b = y1 - m1 * x1
            y = m1 * x + b
        else:

            m1 = (y2 - y1) / (x2 - x1)
            m2 = (y4 - y3) / (x4 - x3)

            if m1 == m2:
                return None, "error"

            a = m1
            b = y1 - m1 * x1
            c = m2
            d = y3 - m2 * x3
            x = (d - b) / (a - c)
            y = a * x + b

        def is_within(x1, x2, x):
            return min(x1, x2) <= x <= max(x1, x2)

        if (
            is_within(x1, x2, x)
            and is_within(y1, y2, y)
            and is_within(x3, x4, x)
            and is_within(y3, y4, y)
        ):
            return (x, y), "intersect"
        return None, "error"


    def find_intersections_2(points, point_pairs):

        intersections = []
        for item in itertools.combinations(point_pairs, 2):
            x1, y1 = points[item[0][0]]
            x2, y2 = points[item[0][1]]
            x3, y3 = points[item[1][0]]
            x4, y4 = points[item[1][1]]
            tmp = find_intersection_3(x1, y1, x2, y2, x3, y3, x4, y4)
            if tmp[1] == "intersect":
                intersections.append(tmp[0])

        filtered_intersections = [point for point in intersections if point not in points]

        return filtered_intersections


    def a2n(letter):
        return ord(letter) - ord("A")


    def a2n2(line):
        return (a2n(line[0]), a2n(line[1]))


    def find_intersection_line_with_segment(x1, y1, x2, y2, x3, y3, x4, y4):
        def is_within(x1, x2, x):
            return min(x1, x2) <= x <= max(x1, x2)

        ans = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
        output = False
        if (
            ans[1] == "intersect"
            and is_within(x3, x4, ans[0][0])
            and is_within(y3, y4, ans[0][1])
        ):
            output = True
        return output


    def polygon_area(points):
        n = len(points)
        area = Fraction(0)
        for i in range(n - 1):
            area += points[i][0] * points[i + 1][1] - points[i][1] * points[i + 1][0]
        area += points[-1][0] * points[0][1] - points[-1][1] * points[0][0]
        return abs(area) / 2


    def surrounding_angle(given_point):
        def is_enclosed_angle(curr, h1, h2, h3):
            return find_intersection_line_with_segment(
                curr[0], curr[1], h2[0], h2[1], h1[0], h1[1], h3[0], h3[1]
            )

        lst = []
        for line in point_pairs:
            if given_point == line[0]:
                lst.append(points[line[1]])
            elif given_point == line[1]:
                lst.append(points[line[0]])

        for item in itertools.permutations(lst):
            if all(
                is_enclosed_angle(points[given_point], item[i], item[i + 1], item[i + 2])
                for i in range(0, len(item) - 2, 1)
            ):
                lst = list(item)
                break

        tmp = [points.index(x) for x in lst]

        return tmp


    def n2a(number):
        return chr(number + ord("A"))


    def straight_line_2(point_list):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles
        point_list = [a2n(x) for x in point_list]
        tmp = polygon_area([points[x] for x in point_list])
        return tmp == Fraction(0)


    def straight_line(point_list):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles
        tmp = polygon_area([points[x] for x in point_list])

        return tmp == Fraction(0)


    def draw_points_and_lines(
        points,
        lines,
        image_size=(1500, 1500),
        point_radius=5,
        point_color=(0, 0, 0),
        line_color=(255, 0, 0),
        text_color=(0, 0, 0),
    ):

        image = Image.new("RGB", image_size, color="white")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except IOError:
            font = ImageFont.load_default()

        for index, (x, y) in enumerate(points):
            draw.ellipse(
                (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                fill=point_color,
            )

            draw.text(
                (x + point_radius + 5, y - point_radius - 5),
                n2a(index),
                fill=text_color,
                font=font,
            )

        for (x1, y1), (x2, y2) in lines:
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=2)

        return image


    def plotter_text(text, color="black", force=False, auto_next_line=True):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles
        return


    def display_image(image_path):
        pass

    def plotter_diagram():
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        image = draw_points_and_lines(points, lines)
        return image


    def travel_till_end(start, step):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles
        done = False
        step_taken = [step]
        while not done:
            done = True
            for item in surrounding_angle(step):
                if (
                    straight_line([step, start, item])
                    and item not in step_taken
                    and start != item
                    and step != start
                    and step != item
                ):
                    step_taken.append(item)
                    step = item

                    done = False
                    break
        return step


    def sur(angle):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles
        count = 0
        if a2n(angle[0]) in surrounding_angle(a2n(angle[1])):
            count += 1
        if a2n(angle[2]) in surrounding_angle(a2n(angle[1])):
            count += 1
        return count


    def plotter_angle(a, b, c, a_do=True, c_do=True):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        if a_do:
            a = travel_till_end(b, a)
        else:
            a = travel_till_end(b, a)
            a = travel_till_end(b, a)
        if c_do:
            c = travel_till_end(b, c)
        else:
            c = travel_till_end(b, c)
            c = travel_till_end(b, c)

        m, n = sorted([a, c])
        return n2a(m) + n2a(b) + n2a(n)


    def plotter_angle_2(angle, a_do=True, c_do=True):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles
        x = angle
        return plotter_angle(a2n(x[0]), a2n(x[1]), a2n(x[2]), a_do, c_do)


    def plotter_angle_3(angle):
        lst = [
            plotter_angle_2(angle, True, True),
            plotter_angle_2(angle, True, False),
            plotter_angle_2(angle, False, True),
            plotter_angle_2(angle, False, False),
        ]
        return sorted(lst, key=lambda x: sur(x))[0]


    def plotter_angle_4(a, b, c):
        return plotter_angle_3(n2a(a) + n2a(b) + n2a(c))


    def combine(a, b):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles


        a = plotter_angle_3(a)
        b = plotter_angle_3(b)
        if a[1] != b[1]:
            return None
        if len(set(a + b)) != 4:
            return None
        r = a[0] + a[2] + b[0] + b[2]
        r = r.replace([x for x in r if r.count(x) == 2][0], "")
        out = plotter_angle_3(r[0] + b[1] + r[1])

        return out


    def angle_sort(angle):
        if a2n(angle[0]) > a2n(angle[2]):
            angle = angle[2] + angle[1] + angle[0]
        return angle


    def line_sort(line):
        if a2n(line[0]) > a2n(line[1]):
            line = line[1] + line[0]
        return line


    all_angles = []


    def break_equation(equation):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        sub_equation_list = [equation]
        equation = equation
        for child in equation.children:
            sub_equation_list += break_equation(child)
        return sub_equation_list


    def row_swap(matrix, i, j):

        matrix[i], matrix[j] = matrix[j], matrix[i]


    def row_scale(matrix, i, scale_factor):

        matrix[i] = [scale_factor * elem for elem in matrix[i]]


    def row_addition(matrix, src_row, dest_row, scale_factor):

        matrix[dest_row] = [
            elem_dest + scale_factor * elem_src
            for elem_dest, elem_src in zip(matrix[dest_row], matrix[src_row])
        ]


    def remove_zero_rows(matrix):

        return [row for row in matrix if any(element != Fraction(0) for element in row)]


    def gauss_jordan_elimination(matrix):

        rows = len(matrix)
        cols = len(matrix[0])
        row = 0

        for col in range(cols):

            pivot_row = max(range(row, rows), key=lambda r: abs(matrix[r][col]))

            if matrix[pivot_row][col] == 0:
                continue

            if pivot_row != row:
                row_swap(matrix, row, pivot_row)

            row_scale(matrix, row, Fraction(1, matrix[row][col]))

            for r in range(rows):
                if r != row:
                    row_addition(matrix, row, r, -matrix[r][col])

            row += 1
            if row >= rows:
                break

        return remove_zero_rows(matrix)


    def matrix_to_list(matrix):
        return [list(item) for item in matrix]


    def fix_matrix():
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        for i in range(len(matrix)):
            if len(matrix[i]) < len(all_angles):
                matrix[i] += [Fraction(0)] * (len(all_angles) - len(matrix[i]))


    def fix_line_matrix():
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles


        target_size = len(line_counter)

        for i in range(len(line_matrix)):

            if len(line_matrix[i]) < target_size:
                line_matrix[i] += [Fraction(0)] * (target_size - len(line_matrix[i]))

            elif len(line_matrix[i]) > target_size:
                line_matrix[i] = line_matrix[i][:target_size]


    def try_line_matrix():
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        if line_matrix == []:
            return

        A = np.array(line_matrix, dtype=object)
        new_matrix = gauss_jordan_elimination(A)

        for item in itertools.combinations(line_counter_convert(), 2):
            row = [Fraction(0)] * (len(line_counter))
            row[index_line(*item[0])] = Fraction(1)
            row[index_line(*item[1])] = Fraction(-1)
            old_matrix = copy.deepcopy(line_matrix)
            line_matrix.append(row)

            if matrices_equal(
                gauss_jordan_elimination(np.array(line_matrix, dtype=object)), new_matrix
            ):

                pass
            else:
                line_matrix = copy.deepcopy(old_matrix)
        fix_line_matrix()


    def matrices_equal(matrix1, matrix2):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            return False

        for row1, row2 in zip(matrix1, matrix2):
            if any(x != y for x, y in zip(row1, row2)):
                return False

        return True


    def try_matrix():
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        if matrix == []:
            return

        A = np.array(matrix, dtype=object)
        B = np.array(matrix_eq, dtype=object)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        augmented_matrix = np.hstack((A, B))

        new_matrix = gauss_jordan_elimination(augmented_matrix)

        for item in itertools.combinations(all_angles, 2):
            row = [Fraction(0)] * len(matrix[0])
            row[all_angles.index(item[0])] = Fraction(1)
            row[all_angles.index(item[1])] = Fraction(-1)
            matrix.append(row)
            matrix_eq.append(Fraction(0))

            if matrices_equal(
                gauss_jordan_elimination(
                    np.hstack(
                        (
                            np.array(matrix, dtype=object),
                            np.array(matrix_eq, dtype=object).reshape(-1, 1),
                        )
                    )
                ),
                new_matrix,
            ):

                pass
            else:
                matrix.pop(-1)
                matrix_eq.pop(-1)
        for item in matrix_to_list(new_matrix):

            non_zero_count = sum(1 for x in item[:-1] if x != Fraction(0))
            if (
                non_zero_count == 1
                and item.count(Fraction(1)) == 1
                and len(item) == len(item[:-1]) + 1
            ):

                matrix.append(item[:-1])
                matrix_eq.append(item[-1])


    def line_matrix_plotter(plotter_it=True):
        def remove_duplicate_rows(matrix):

            unique_rows = set(tuple(row) for row in matrix)

            return [list(row) for row in unique_rows]

        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles


        fix_line_matrix()

        line_matrix = remove_duplicate_rows(line_matrix)

        string = "$"
        for i in range(len(line_matrix)):
            for j in range(len(line_matrix[i])):
                if line_matrix[i][j] != Fraction(0):
                    if line_matrix[i][j] < Fraction(0):
                        string += "-line(" + index_line_matrix(j) + ")"
                    else:
                        string += "+line(" + index_line_matrix(j) + ")"
            string += "=0\n"
        string = string.replace("\n+", "\n").replace("$+", "").replace("$", "")
        if plotter_it and string != "":
            plotter_text(string, "black", False, False)


    def matrix_plotter(plotter_it=True):
        def remove_duplicates(matrix_2d, array_1d):
            unique_rows = {}
            for row, val in zip(matrix_2d, array_1d):
                row_tuple = tuple(row)
                if row_tuple not in unique_rows:
                    unique_rows[row_tuple] = val
            new_matrix_2d = list(unique_rows.keys())
            new_array_1d = list(unique_rows.values())
            return new_matrix_2d, new_array_1d

        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        fix_matrix()
        matrix, matrix_eq = remove_duplicates(matrix, matrix_eq)
        for i in range(len(matrix)):
            matrix[i] = list(matrix[i])

        string = "$"
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] != Fraction(0):
                    if matrix[i][j] == Fraction(-1):
                        string += "-angle(" + all_angles[j] + ")"
                    elif matrix[i][j] == Fraction(1):
                        string += "+angle(" + all_angles[j] + ")"
                    else:
                        string += (
                            "+" + str(float(matrix[i][j])) + "*angle(" + all_angles[j] + ")"
                        )
            string += "=" + str(int(matrix_eq[i])) + "\n"
        string = string.replace("\n+", "\n").replace("$+", "").replace("$", "")
        if plotter_it and string != "":
            plotter_text(string, "black", False, False)


    def process(plotter_it=True):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles


        lines = [(points[start], points[end]) for start, end in point_pairs]
        plotter_diagram()
        output = []
        find = find_intersections_2(points, point_pairs)

        for i in range(len(points)):

            for angle in itertools.combinations(surrounding_angle(i), 2):

                if angle[0] != angle[1]:

                    output.append(plotter_angle_4(angle[0], i, angle[1]))

        output = list(set(output))
        append_angles = set(output) - set(all_angles)
        reject_angles = set(all_angles) - set(output)

        all_angles = all_angles + list(append_angles)

        index_iter = sorted([all_angles.index(item) for item in reject_angles])[::-1]

        for i in range(len(matrix) - 1, -1, -1):
            for item in index_iter:
                if int(matrix[i][item]) != Fraction(0):
                    matrix.pop(i)
                    matrix_eq.pop(i)
                    break
        for item in index_iter:
            for i in range(len(matrix)):
                matrix[i].pop(item)
            all_angles.pop(item)
        for item in point_pairs:
            s = n2a(item[0]) + n2a(item[1])
            index_line(*s)
        output = []
        
        generate_all_lines()
        fix_line_matrix()
        
        matrix_plotter(plotter_it)
        line_matrix_plotter(plotter_it)
        eq_list = list(set(eq_list))
        


    def all_cycle(graph):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        cycles = []

        def findNewCycles(path):
            start_node = path[0]
            next_node = None
            for edge in graph:
                node1, node2 = edge
                if start_node in edge:
                    next_node = node2 if node1 == start_node else node1
                    if not visited(next_node, path):
                        sub_path = [next_node] + path
                        findNewCycles(sub_path)
                    elif len(path) > 2 and next_node == path[-1]:
                        p = rotate_to_smallest(path)
                        inv = invert(p)
                        if isNew(p) and isNew(inv):
                            cycles.append(p)

        def invert(path):
            return rotate_to_smallest(path[::-1])

        def rotate_to_smallest(path):
            n = path.index(min(path))
            return path[n:] + path[:n]

        def isNew(path):
            return path not in cycles

        def visited(node, path):
            return node in path

        for edge in graph:
            for node in edge:
                findNewCycles([node])
        return cycles


    def perpendicular_line_intersection(segment_start, segment_end, point):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        x1, y1 = segment_start
        x2, y2 = segment_end
        xp, yp = point

        if x2 == x1:
            xq = x1
            yq = yp

        elif y2 == y1:
            xq = xp
            yq = y1

        else:

            m = (y2 - y1) / (x2 - x1)

            m_perp = -1 / m

            xq = (m * x1 - m_perp * xp + yp - y1) / (m - m_perp)

            yq = m * (xq - x1) + y1

        return (xq, yq)


    def is_reflex_vertex(polygon, vertex_index):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        prev_index = (vertex_index - 1) % len(polygon)
        next_index = (vertex_index + 1) % len(polygon)
        modified_polygon = polygon[:vertex_index] + polygon[vertex_index + 1 :]
        original_area = polygon_area(polygon)
        modified_area = polygon_area(modified_polygon)
        if modified_area <= original_area:
            return False
        else:
            return True


    def is_reflex_by_circle(polygon):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        output = []
        for i in range(len(polygon)):
            if is_reflex_vertex(polygon, i):
                output.append(i)
        return output


    def all_triangle(all_angles):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter


        cycle = all_cycle(point_pairs)

        new_cycle = []
        for item in cycle:
            remove_item = []
            for i in range(-2, len(item) - 2, 1):
                if straight_line([item[i], item[i + 1], item[i + 2]]):
                    remove_item.append(item[i + 1])
            new_item = item
            for i in range(len(new_item) - 1, -1, -1):
                if new_item[i] in remove_item:
                    new_item.pop(i)
            new_cycle.append(new_item)
        for x in new_cycle:

            to_remove = []
            for i in range(-2, len(x) - 2, 1):
                angle = [x[i], x[i + 1], x[i + 2]]
                if straight_line(angle):
                    to_remove.append(i)
            to_remove = sorted(to_remove)[::-1]

            for item in to_remove:
                x.pop(item)

            convex_angle = is_reflex_by_circle([points[y] for y in x])

            out = []
            v = None
            for i in range(-2, len(x) - 2, 1):
                angle = [x[i], x[i + 1], x[i + 2]]
                tmp = [[z for z in x][y] for y in convex_angle]

                v = "".join([n2a(y) for y in angle])
                if angle[1] in tmp:
                    out.append(
                        "(360-" + plotter_angle_3("".join([n2a(y) for y in angle])) + ")"
                    )
                else:
                    out.append(plotter_angle_3("".join([n2a(y) for y in angle])))

            if len(x) == 3:
                all_tri.append(v)
            if out == []:
                continue

            copy_out = copy.deepcopy(out)

            out = copy.deepcopy(copy_out)
            for i in range(len(out)):
                out[i] = out[i].replace("(360-", "").replace(")", "")

            matrix.append([Fraction(0)] * len(all_angles))
            subtract = 0

            for i in range(len(out)):
                if "(360-" in copy_out[i]:

                    subtract += 360
                    matrix[-1][all_angles.index(out[i])] = Fraction(-1)
                else:
                    matrix[-1][all_angles.index(out[i])] = Fraction(1)
            matrix_eq.append(Fraction(180 * (len(x) - 2) - subtract))

        all_tri = list(set(all_tri))


    def fix_angle_line(eq):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles


        eq = tree_form(eq)
        content = list(set("".join(all_angles)))
        for item2 in itertools.permutations(content, 3):
            item2 = "".join(item2)
            if item2 not in string_equation(str_form(eq)):
                continue

            eq = replace(eq, convert_angle_2(item2), convert_angle_2(plotter_angle_3(item2)))
            eq = replace(
                eq,
                tree_form(str_form(convert_angle_2(item2)).replace("f_angle", "f_xangle")),
                tree_form(
                    str_form(convert_angle_2(plotter_angle_3(item2))).replace(
                        "f_angle", "f_xangle"
                    )
                ),
            )
        for item in point_pairs:
            item = n2a(item[0]) + n2a(item[1])
            eq = replace(eq, line_fx(item), line_fx(line_sort(item)))
            eq = replace(eq, line_fx(item[1] + item[0]), line_fx(line_sort(item)))
            eq = replace(
                eq,
                tree_form(str_form(line_fx(item)).replace("f_line", "f_xline")),
                tree_form(str_form(line_fx(line_sort(item))).replace("f_line", "f_xline")),
            )
            eq = replace(
                eq,
                tree_form(
                    str_form(line_fx(item[1] + item[0])).replace("f_line", "f_xline")
                ),
                tree_form(str_form(line_fx(line_sort(item))).replace("f_line", "f_xline")),
            )
        return str_form(eq)


    def extend(line, point_start, distance):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        b = None
        a = points[a2n(point_start)]
        if line[0] == point_start:
            b = points[a2n(line[1])]
        else:
            b = points[a2n(line[0])]
        ba = [a[0] - b[0], a[1] - b[1]]
        length_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        unit_vector_ba = [ba[0] / length_ba, ba[1] / length_ba]
        bc = [unit_vector_ba[0] * distance, unit_vector_ba[1] * distance]
        c = tuple([Fraction(round(a[0] + bc[0])), Fraction(round(a[1] + bc[1]))])
        out = c
        if polygon_area([a, b, c]) != Fraction(0):
            out = perpendicular_line_intersection(a, b, c)
        points.append(out)
        plotter_text("new point added")


    def divide_line(line, new_val=None):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        a = a2n(line[0])
        b = a2n(line[1])
        if (a, b) not in point_pairs:
            a, b = b, a
            if (a, b) not in point_pairs:
                return None
        new_point = None
        if new_val is None:
            new_point = (
                round((points[a][0] + points[b][0]) / 2),
                round((points[a][1] + points[b][1]) / 2),
            )
        else:
            new_point = new_val

        point_pairs.pop(point_pairs.index((a, b)))
        point_pairs.append((len(points), a))
        point_pairs.append((len(points), b))
        points.append((Fraction(new_point[0]), Fraction(new_point[1])))


    def is_point_on_line(line, point):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        a = points[line[0]]
        b = points[line[1]]
        c = point
        
        
        return polygon_area([a, b, c]) == Fraction(0)

    def is_point_on_line_2(line, point):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        a = points[line[0]]
        b = points[line[1]]
        c = point
        def is_within(x1, x2, x):
            return min(x1, x2) <= x <= max(x1, x2)
        return is_within(a[0], b[0], c[0]) and is_within(a[1], b[1], c[1]) and polygon_area([a, b, c]) == Fraction(0)

    def find_line_for_point(point):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        output = []
        for i, line in enumerate(point_pairs):
            if is_point_on_line(line, point):
                output.append(i)
        return output


    def connect_point(point_ab):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        output = []
        point_a, point_b = point_ab
        point_pairs.append((a2n(point_a), a2n(point_b)))

        inter = find_intersections_2(points, point_pairs)

        for p in inter:

            item_list = find_line_for_point(p)

            points.append(p)
            to_remove = []
            to_add = []
            for item in item_list:
                a, b = point_pairs[item]
                to_remove.append(point_pairs.index((a, b)))
                to_add.append((len(points) - 1, a))
                to_add.append((len(points) - 1, b))
            a1, a2, a3, a4 = to_add[0][1], to_add[1][1], to_add[2][1], to_add[3][1]
            s1 = plotter_angle_4(a1, len(points) - 1, a4), plotter_angle_4(
                a3, len(points) - 1, a2
            )
            s2 = plotter_angle_4(a1, len(points) - 1, a3), plotter_angle_4(
                a4, len(points) - 1, a2
            )

            output.append(s1)

            output.append(s2)

            to_remove = sorted(to_remove)[::-1]
            for item in to_remove:
                point_pairs.pop(item)
            for item in to_add:
                point_pairs.append(item)

        return output


    def draw_triangle():
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        points = [
            (Fraction(400), Fraction(800)),
            (Fraction(800), Fraction(750)),
            (Fraction(600), Fraction(400)),
        ]

        point_pairs = [(0, 1), (1, 2), (2, 0)]


    def perpendicular(point, line, ver=1):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        output = None
        if point in line or straight_line_2([point] + list(line)):
            return "command parsing error"
        if ver == 1:
            output = perpendicular_line_intersection(
                points[a2n(line[0])], points[a2n(line[1])], points[a2n(point)]
            )
        else:
            output = perpendicular_line_intersection2(
                points[a2n(line[0])], points[a2n(line[1])], points[a2n(point)]
            )
        if all(not is_point_on_line_2([a2n(line_sort(x)[0]), a2n(line_sort(x)[1])],output) for x in line_counter_convert()):
            return "command parsing error"
        divide_line(line, output)
        num = len(points) - 1

        tmp = connect_point(n2a(len(points) - 1) + point)
        process(False)
        for item in tmp:
            row = [Fraction(0)] * len(all_angles)
            row[all_angles.index(item[0])] = Fraction(1)
            row[all_angles.index(item[1])] = Fraction(-1)
            matrix.append(row)
            matrix_eq.append(Fraction(0))

        eq1 = plotter_angle_3(point + n2a(num) + line[0])
        eq2 = plotter_angle_3(point + n2a(num) + line[1])
        row = [Fraction(0)] * len(all_angles)
        row[all_angles.index(eq1)] = Fraction(1)
        matrix_eq.append(Fraction(90))
        matrix.append(row)
        row = [Fraction(0)] * len(all_angles)
        row[all_angles.index(eq2)] = Fraction(1)
        matrix_eq.append(Fraction(90))
        matrix.append(row)
        return None

    def load_image(file_path):
        pass


    def convert_form(eq):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        def num_not_var(eq):
            for index, item in enumerate(all_angles):
                eq = replace(eq, convert_angle(item), tree_form("v_" + str(index)))
            return eq

        if all(
            x not in eq
            for x in [
                "f_triangle",
                "f_xtriangle",
                "f_xangle",
                "f_congruent",
                "f_xcongruent",
                "f_line",
                "f_xline",
            ]
        ):
            return str_form(num_not_var(tree_form(eq)))
        return None


    def convert_form_2(eq):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles


        def num_not_var(eq):
            for index, item in enumerate(line_counter):
                eq = replace(
                    eq, line_fx(index_line_matrix(index)), tree_form("v_" + str(index))
                )
            return eq

        if all(
            x not in eq
            for x in [
                "f_triangle",
                "f_xtriangle",
                "f_xangle",
                "f_angle",
                "f_congruent",
                "f_xcongruent",
                "f_xline",
            ]
        ):
            return str_form(num_not_var(tree_form(eq)))
        return None


    def find_all_paths(graph, start_node, end_node, path=[]):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        path = path + [start_node]
        if start_node == end_node:
            return [path]
        if start_node not in graph:
            return []
        paths = []
        for neighbor in graph[start_node]:
            if neighbor not in path:
                new_paths = find_all_paths(graph, neighbor, end_node, path)
                for p in new_paths:
                    paths.append(p)
        return paths


    def generate_graph():
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        graph = dict()
        for i in range(len(points)):
            graph[n2a(i)] = [n2a(x) for x in surrounding_angle(i)]
        return graph


    def generate_all_lines():
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        for item in itertools.combinations(range(len(points)), 2):
            for path in find_all_paths(generate_graph(), item[0], item[1]):
                if straight_line_2(path):
                    for item2 in itertools.combinations(path, 2):
                        index_line(item2[0], item2[1])


    def is_same_line(line1, line2):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        if line1 == line2:
            return True
        for item in itertools.combinations(range(len(points)), 2):
            for path in find_all_paths(generate_graph(), item[0], item[1]):
                if (
                    straight_line_2(path)
                    and line1[0] in path
                    and line1[1] in path
                    and line2[0] in path
                    and line2[1] in path
                ):
                    
                    return True
        
        return False


    def do_isoceles():
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        lst = line_counter_convert()
        for item in line_matrix:
            if (
                item.count(Fraction(0)) == len(item) - 2
                and item.count(Fraction(1)) == 1
                and item.count(Fraction(-1)) == 1
            ):
                line1 = lst[item.index(1)]
                line2 = lst[item.index(-1)]
                for tri in all_tri:
                    if (
                        line1[0] in tri
                        and line2[0] in tri
                        and line1[1] in tri
                        and line2[1] in tri
                    ):
                        common = set(line1) & set(line2)
                        common = list(common)[0]
                        a = list(set(line1) - set(common))[0]
                        b = list(set(line2) - set(common))[0]
                        row = [Fraction(0)] * len(all_angles)
                        row[all_angles.index(plotter_angle_3(common + a + b))] = Fraction(1)
                        row[all_angles.index(plotter_angle_3(common + b + a))] = Fraction(-1)
                        matrix.append(row)
                        matrix_eq.append(Fraction(0))
                        break


    def do_cpct():
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        cpct_output = []
        for item in eq_list:
            if "congruent" in item:
                m = re.findall(r"[A-Z]{3}", string_equation(item))
                m_list = []
                for item2 in itertools.permutations(range(3)):
                    m_list.append(
                        [
                            m[0][item2[0]] + m[0][item2[1]] + m[0][item2[2]],
                            m[1][item2[0]] + m[1][item2[1]] + m[1][item2[2]],
                        ]
                    )
                for item2 in m_list:
                    add_angle_equality(item2[0], item2[1])
                    add_line_equality(item2[0][:-1], item2[1][:-1])
        fix_matrix()

    def process_join(join_arr):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles

        for join_iter in join_arr:
            tmp = connect_point(join_iter)
        process(False)
        for item in tmp:
            row = [Fraction(0)] * len(all_angles)
            row[all_angles.index(item[0])] = Fraction(1)
            row[all_angles.index(item[1])] = Fraction(-1)
            matrix.append(row)
            matrix_eq.append(Fraction(0))

    def walk_parallel_function(command):
        nonlocal points, point_pairs, eq_list, plotter_on, matrix, matrix_eq, all_tri, line_matrix, lines, line_counter, all_angles
        
        plotter_on = True
        line_counter = []
        all_tri = []
        points = []
        point_pairs = []
        eq_list = []
        lines = []
        matrix = []
        matrix_eq = []
        line_matrix = []
        
        try:
            orig_command = command
            join_arr = []
            command = command.split("\n")
            string = None
            index = 0
            join_dont_plotter = True
            while True:
                if string not in {"show", "hide"}:
                    process(join_dont_plotter)
                if command != []:
                    string = command.pop(0)
                    index += 1
                    plotter_text(">>> ", "green", True, False)
                    plotter_text(string, "blue", True, True)
                else:
                    if join_arr != []:
                        process_join(join_arr)
                        process()
                    plotter_text("\nend of program", "green", True, True)
                    return plotter_diagram(), len(points), line_counter_convert(), all_angles
                if string.split(" ")[0] != "join" and join_arr != []:
                    process_join(join_arr)
                    join_arr = []
                    join_dont_plotter = True
                if string == "hide":
                    plotter_on = False
                if string == "show":
                    plotter_on = True
                
                if string[:13] == "draw triangle" and index == 1:
                    draw_triangle()
                elif string == "draw quadrilateral" and index == 1:
                    points = [
                        (Fraction(400), Fraction(800)),
                        (Fraction(800), Fraction(750)),
                        (Fraction(600), Fraction(400)),
                        (Fraction(400), Fraction(550)),
                    ]
                    point_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
                elif string == "compute":
                    try_line_matrix()

                    def two(angle1, angle2):
                        list1 = list(itertools.permutations(list(angle1)))
                        list2 = list(itertools.permutations(list(angle2)))
                        for i in range(len(list1)):
                            for j in range(len(list2)):
                                out = proof_fx_3("".join(list1[i]), "".join(list2[j]))
                                if out is not None:

                                    return

                    if len(all_tri) > 1:

                        for item in itertools.combinations(all_tri, 2):

                            two(item[0], item[1])

                    output = try_matrix()
                elif string.split(" ")[0] == "perpendicular" and string.split(" ")[2] == "to":
                    tmp4 = perpendicular(string.split(" ")[1], string.split(" ")[3])
                    if tmp4 is not None:
                        plotter_text("\ncommand parsing error", "green", True, True)
                        return "error"
                elif string == "calculate":
                    output = try_matrix()
                    eq_list = list(set(eq_list))
                elif string.split(" ")[0] == "prove":
                    proof = parser_4.take_input(string.split(" ")[1])
                    output = proof_fx(proof)
                    if output:
                        eq_list.append(output)
                elif (
                    string.split(" ")[0] == "extend"
                    and string.split(" ")[2] == "from"
                    and string.split(" ")[4] == "for"
                ):
                    extend(
                        string.split(" ")[1], string.split(" ")[3], int(string.split(" ")[5])
                    )
                    
                elif string.split(" ")[0] == "extend" and string.split(" ")[2] == "to":
                    val = find_intersection(
                        points[a2n(string.split(" ")[1][0])][0],
                        points[a2n(string.split(" ")[1][0])][1],
                        points[a2n(string.split(" ")[1][1])][0],
                        points[a2n(string.split(" ")[1][1])][1],
                        points[a2n(string.split(" ")[3][0])][0],
                        points[a2n(string.split(" ")[3][0])][1],
                        points[a2n(string.split(" ")[3][1])][0],
                        points[a2n(string.split(" ")[3][1])][1],
                    )
                    divide_line(string.split(" ")[3], val[0])

                elif string.split(" ")[0] == "split":
                    divide_line(string.split(" ")[-1])
                elif string == "cpct":
                    do_cpct()
                elif string.split(" ")[0] == "join" and all(x[0] != x[1] and line_sort(x) not in line_counter_convert() for x in string.split(" ")[1:]) and\
                     len(set([line_sort(x) for x in join_arr + string.split(" ")[1:]])) == len([line_sort(x) for x in join_arr + string.split(" ")[1:]]):
                    list_join = string.split(" ")[1:]
                    join_arr += list_join
                    join_dont_plotter = False
                elif string.split(" ")[0] == "equation":

                    eq_type = string.split(" ")[1]
                    if "angle_eq" == eq_type:
                        pass
                    elif "angle_val" == eq_type:
                        pass
                    elif "parallel_line" == eq_type:
                        pass
                    elif "line_eq" == eq_type:
                        pass
                else:
                    plotter_text("\ncommand parsing error", "green", True, True)
                    return "error"
        except:
            return "error"
    return walk_parallel_function(command)




# Function to create a black-and-white placeholder image
def create_placeholder_image():
    image = Image.new('RGB', (500, 500), color='white')
    draw = ImageDraw.Draw(image)
    draw.text((200, 240), "Placeholder", fill='black')  # Optional placeholder text
    return image

# Initialize session state for accumulated commands and initial image
if 'accumulated_commands' not in st.session_state:
    st.session_state.accumulated_commands = ""  # String to accumulate commands

if 'diagram_image' not in st.session_state:
    # Initialize with a black-and-white placeholder image
    st.session_state.diagram_image = create_placeholder_image()

if 'num_points' not in st.session_state:
    st.session_state.num_points = 0  # To store the number of points for the join command

if 'lines' not in st.session_state:
    st.session_state.lines = []  # To store lines returned by the engine

if 'angles' not in st.session_state:
    st.session_state.angles = []  # To store angles returned by the engine

# Function to walk geometry commands with accumulated command string
def walk_geometry_command(command):
    # Prepare the full command string by appending the new command
    new_accumulated_commands = f"{st.session_state.accumulated_commands}\n{command}".strip()
    result = ENGINE_R(new_accumulated_commands)  # Send the whole string
    if result == "error":
        return None, "error"
    else:
        # Update the accumulated commands only if the command is successful
        st.session_state.accumulated_commands = new_accumulated_commands

        # Assuming result is a tuple: (image, num_points, line list, angle list)
        diagram_image = result[0]  # Image of the diagram
        num_points = result[1]  # Number of points
        line_list = result[2]  # List of lines
        angle_list = result[3]  # List of angles

        # Update session state
        st.session_state.num_points = num_points
        st.session_state.lines = line_list  # Store the lines for further commands
        st.session_state.angles = angle_list  # Store the angles for further commands
        
        return {
            "image": diagram_image,
            "num_points": num_points,
            "line_list": line_list,
            "angle_list": angle_list
        }, None

# Streamlit app interface
st.title("Geometry Ai")

# Sidebar with tool selection
st.sidebar.title("Toolbox")
tool = st.sidebar.radio("Select a tool:", ("Draw", "Perpendicular", "Join", "Extend", "Split", "Set Angles Equal", "Angle Value", "Set Lines Equal", "Parallel Line"))

st.sidebar.subheader("Accumulated Commands")
st.sidebar.text_area("Commands", value=st.session_state.accumulated_commands, height=200)


# Tool: Draw (Triangle or Quadrilateral)
if tool == "Draw":
    draw_option = st.radio("What would you like to draw?", ("Draw Triangle", "Draw Quadrilateral"))
    command = "draw triangle" if draw_option == "Draw Triangle" else "draw quadrilateral"

    if st.button("Run Command"):
        result, error = walk_geometry_command(command)
        if error:
            st.error("Invalid command. Please try again.")
        else:
            if result["image"]:
                st.session_state.diagram_image = result["image"]

            if result["num_points"]:
                st.write(f"Number of Points: {result['num_points']}")
            if result["line_list"]:
                st.write("Line List:")
                st.write(result["line_list"])
            if result["angle_list"]:
                st.write("Angle List:")
                st.write(result["angle_list"])
            

# Tool: Join
if tool == "Join":
    if st.session_state.num_points > 1:
        points = [f"{chr(65 + i)}" for i in range(st.session_state.num_points)]  # Generates ['A', 'B', 'C', ...]
        point1 = st.selectbox("Select the first point to join:", points)
        point2 = st.selectbox("Select the second point to join:", points)

        if st.button("Join Points"):
            if point1 != point2:  # Ensure different points are selected
                command = f"join {point1}{point2}"
                result, error = walk_geometry_command(command)

                if error:
                    st.error("Invalid command. Please try again.")
                else:
                    if result["image"]:
                        st.session_state.diagram_image = result["image"]

                    if result["num_points"]:
                        st.write(f"Number of Points: {result['num_points']}")
                    if result["line_list"]:
                        st.write("Line List:")
                        st.write(result["line_list"])
                    if result["angle_list"]:
                        st.write("Angle List:")
                        st.write(result["angle_list"])
            else:
                st.error("Please select two different points to join.")
    else:
        st.warning("At least two points are needed to perform a join operation.")

# Tool: Extend
if tool == "Extend":
    if st.session_state.num_points > 0 and st.session_state.lines:
        points = [f"{chr(65 + i)}" for i in range(st.session_state.num_points)]  # Points A, B, C, ...
        lines = st.session_state.lines  # List of lines returned by the engine

        # Select a point for extending
        point = st.selectbox("Select a point to extend from:", points)
        
        # Select a line to extend
        line = st.selectbox("Select a line to extend:", lines)

        # Slider for the extension length
        extension_length = st.slider("Extension Length (units):", min_value=1, max_value=200, value=100)

        if st.button("Extend Line"):
            command = f"extend {line} from {point} for {extension_length}"
            result, error = walk_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least one point and one line are needed to perform an extend operation.")

# Tool: Perpendicular
if tool == "Perpendicular":
    if st.session_state.lines:
        lines = st.session_state.lines  # List of lines returned by the engine
        line_to_perpendicular = st.selectbox("Select a line to create a perpendicular:", lines)

        # Select a point for the perpendicular
        points = [f"{chr(65 + i)}" for i in range(st.session_state.num_points)]  # Points A, B, C, ...
        point = st.selectbox("Select a point to create the perpendicular from:", points)

        if st.button("Create Perpendicular"):
            command = f"perpendicular {point} to {line_to_perpendicular}"
            result, error = walk_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least one line is needed to perform a perpendicular operation.")


# Tool: Split
if tool == "Split":
    if st.session_state.lines:
        lines = st.session_state.lines  # List of lines returned by the engine
        line_to_split = st.selectbox("Select a line to split:", lines)

        if st.button("Split Line"):
            command = f"split {line_to_split}"
            result, error = walk_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least one line is needed to perform a split operation.")

# Tool: Set Angles Equal
if tool == "Set Angles Equal":
    if st.session_state.angles and len(st.session_state.angles) > 1:
        angles = [f"∠{angle}" for angle in st.session_state.angles]  # Formatting angles
        angle1 = st.selectbox("Select the first angle to set equal:", angles)[1:]
        angle2 = st.selectbox("Select the second angle to set equal:", angles)[1:]

        if st.button("Set Angles Equal"):
            if angle1 != angle2:  # Ensure different angles are selected
                command = f"equation angle_eq {angle1} {angle2}"
                result, error = walk_geometry_command(command)

                if error:
                    st.error("Invalid command. Please try again.")
                else:
                    if result["image"]:
                        st.session_state.diagram_image = result["image"]

                    if result["num_points"]:
                        st.write(f"Number of Points: {result['num_points']}")
                    if result["line_list"]:
                        st.write("Line List:")
                        st.write(result["line_list"])
                    if result["angle_list"]:
                        st.write("Angle List:")
                        st.write(result["angle_list"])
            else:
                st.error("Please select two different angles to set equal.")
    else:
        st.warning("At least two angles are needed to perform the set equal operation.")

# Tool: Angle Value
if tool == "Angle Value":
    if st.session_state.angles and len(st.session_state.angles) > 0:
        angles = [f"∠{angle}" for angle in st.session_state.angles]  # Formatting angles
        angle = st.selectbox("Select an angle to set value:", angles)[1:]

        angle_value = st.number_input("Enter the angle value (degrees):", min_value=0, max_value=360)

        if st.button("Set Angle Value"):
            command = f"equation angle_val {angle} {angle_value}"
            result, error = walk_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least two lines are needed to perform the set equal operation.")

# Tool: Line Equality
if tool == "Set Lines Equal":
    if st.session_state.lines and len(st.session_state.lines) > 1:
        lines = st.session_state.lines  # List of lines returned by the engine
        line1 = st.selectbox("Select the first line:", lines)
        line2 = st.selectbox("Select the second line:", lines)

        if st.button("Set Lines Equal"):
            command = f"equation line_eq {line1} {line2}"  # Assuming this command checks if lines are equal
            result, error = walk_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least two lines are needed to check equality.")



# Tool: Parallel Line
if tool == "Parallel Line":
    if st.session_state.lines and st.session_state.num_points > 0:
        lines = st.session_state.lines  # List of lines returned by the engine
        line1 = st.selectbox("Select the first line (for reference):", lines)
        line2 = st.selectbox("Select the second line (to be parallel to):", lines)

        if st.button("Create Parallel Line"):
            command = f"equation parallel_line {line2} {line1}"
            result, error = walk_geometry_command(command)

            if error:
                st.error("Invalid command. Please try again.")
            else:
                if result["image"]:
                    st.session_state.diagram_image = result["image"]

                if result["num_points"]:
                    st.write(f"Number of Points: {result['num_points']}")
                if result["line_list"]:
                    st.write("Line List:")
                    st.write(result["line_list"])
                if result["angle_list"]:
                    st.write("Angle List:")
                    st.write(result["angle_list"])
    else:
        st.warning("At least one line is needed to create a parallel line.")

# Add button to walk accumulated commands
if st.sidebar.button("Execute Accumulated Commands"):
    command = st.session_state.accumulated_commands  # Get the accumulated commands
    if command:  # Ensure there are commands to execute
        output = ENGINE(command + "\ncompute\ncompute")  # walk the accumulated commands
        st.subheader("Analysis:")
        st.markdown("```\n" + output + "\n```") 
    else:
        st.warning("No commands to execute.")

# Display the diagram image
st.image(st.session_state.diagram_image, caption="Geometry Diagram", use_column_width=True)
