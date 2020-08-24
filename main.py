import random
import numpy as np
import matplotlib.pyplot as plt
import math


def generate_dots(number_of_dots, side_lenght):
    '''Generate dots of the polygon

    Для генерации точек используется псевдолинейное распределение
    '''
    all_x = []
    all_y = []
    for i in range(number_of_dots):
        angle_1 = 360 * i / number_of_dots
        angle_2 = 360 * (i + 1) / number_of_dots

        dot_angle = np.random.uniform(angle_1, angle_2, 1)

        boundary = min(side_lenght, abs(side_lenght / math.tan(math.radians(dot_angle))))

        dot_x = (np.random.uniform(0, boundary, 1)[0] * np.sign(math.sin(math.radians(dot_angle)))
                 * np.sign(math.tan(math.radians(dot_angle))))

        dot_y = (dot_x * (math.tan(math.radians(dot_angle)))
                 * np.sign(math.sin(math.radians(dot_angle))) / np.sign(math.sin(math.radians(dot_angle))))

        check_value = np.random.random(1)[0] * 1.8

        while check_value > dot_x ** 2 + dot_y ** 2:
            dot_x = (np.random.uniform(0, boundary, 1)[0] * np.sign(math.sin(math.radians(dot_angle)))
                     * np.sign(math.tan(math.radians(dot_angle))))
            dot_y = (dot_x * (math.tan(math.radians(dot_angle)))
                     * np.sign(math.sin(math.radians(dot_angle))) / np.sign(math.sin(math.radians(dot_angle))))

            check_value = np.random.random(1)[0] * 1.8

        all_x.append(dot_x)
        all_y.append(dot_y)
    return all_x, all_y


def drawPolygon():
    """
    Рисование многоугольника
    """
    coord = [(all_x[i], all_y[i]) for i in range(len(all_x))]
    coord.append(coord[0])  # repeat the first point to create a 'closed loop'

    xs, ys = zip(*coord)  # create lists of x and y values

    plt.fill(xs, ys, alpha=0.3)


def splitPolygon():
    #     for i in range (len(all_angles)):
    #         if all_angles[i]>=180:
    #             print ('выпуклый угол:', i)

    coord = [(all_x[i], all_y[i]) for i in range(len(all_x))]

    mydict = {i: coord[i] for i in range(len(coord))}

    coord.append(coord[0])  # repeat the first point to create a 'closed loop'
    #    coord.append(coord[1])
    divisible_polygons = []
    polygons_angles = []

    divisible_polygons.append(coord)

    zip_angles = []
    for i in range(len(coord)):
        zip_angles.append(list(mydict.keys())[list(mydict.values()).index(coord[i])])
    polygons_angles.append(zip_angles)

    return divisible_polygons


def drawTriangle(divisible_polygons):
    """
    Рисование треугольника
    """
    all_triangles = []
    for i in range(len(divisible_polygons)):
        for j in range(len(divisible_polygons[i]) - 1):
            coord = ([divisible_polygons[i][j], (0, 0), divisible_polygons[i][j + 1]])
            coord_copy = coord.copy()
            all_triangles.append(coord_copy)

            coord.append(coord[0])  # repeat the first point to create a 'closed loop'
            xs, ys = zip(*coord)  # create lists of x and y values
            plt.plot(xs, ys, color='black', alpha=1)
    return all_triangles


def genBasis():
    basis = []
    basis_angles = []
    for length in np.linspace(0.3, 1, 5):
        for angle in np.linspace(0, 180, 40):
            if angle not in basis_angles:
                basis_angles.append(angle)
            coord = ([(0, 0), (length, 0), (np.cos(math.radians(angle)), np.sin(math.radians(angle)))])
            tmp_coord = coord.copy()
            basis.append(tmp_coord)

            coord.append(coord[0])  # repeat the first point to create a 'closed loop'
            xs, ys = zip(*coord)  # create lists of x and y values
            # plt.plot (xs,ys, color='black', alpha = 1)

    # print(basis)
    # print(all_angles)
    return basis, basis_angles


'''нахождение токи пересечения 2 прямых'''


def line(p1, p2):
    '''Создание прямой A*x +B*y +(-?)C = 0
    '''
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    '''Нахождение точки пересечения прямых L1, L2
    L1, L2 - массивы значений из line()'''
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


''' '''


def split_triangles_into_shorter_ones(all_triangles):
    '''Разбиение секторов на треугольники, чьи размеры не превышают 1
    UPD: стоит пофиксить, что необходимо наличие лишь двух сторон <=1, а не трех'''
    for i in range(len(all_triangles)):
        tmp_triangles = [all_triangles[i]]

        while tmp_triangles != []:
            len_side_1 = (np.sqrt((tmp_triangles[0][0][0] - tmp_triangles[0][1][0]) ** 2 +
                                  (tmp_triangles[0][0][1] - tmp_triangles[0][1][1]) ** 2))
            len_side_2 = (np.sqrt((tmp_triangles[0][2][0] - tmp_triangles[0][1][0]) ** 2 +
                                  (tmp_triangles[0][2][1] - tmp_triangles[0][1][1]) ** 2))
            len_side_3 = (np.sqrt((tmp_triangles[0][0][0] - tmp_triangles[0][2][0]) ** 2
                                  + (tmp_triangles[0][0][1] - tmp_triangles[0][2][1]) ** 2))

            if len_side_1 > 1 and len_side_1 > len_side_2 and len_side_1 > len_side_3:
                tmp_triangles.append([((tmp_triangles[0][0][0] + tmp_triangles[0][1][0]) / 2,
                                       (tmp_triangles[0][0][1] + tmp_triangles[0][1][1]) / 2),
                                      tmp_triangles[0][1], tmp_triangles[0][2]])
                tmp_triangles.append([tmp_triangles[0][0],
                                      ((tmp_triangles[0][0][0] + tmp_triangles[0][1][0]) / 2,
                                       (tmp_triangles[0][0][1] + tmp_triangles[0][1][1]) / 2),
                                      tmp_triangles[0][2]])

                '''draw line'''
                line_coord = ([((tmp_triangles[0][0][0] + tmp_triangles[0][1][0]) / 2,
                                (tmp_triangles[0][0][1] + tmp_triangles[0][1][1]) / 2),
                               tmp_triangles[0][2]])
                line_x, line_y = zip(*line_coord)  # create lists of x and y values
                plt.plot(line_x, line_y, color='black', alpha=1)
                tmp_triangles.remove(tmp_triangles[0])


            elif len_side_2 > 1 and len_side_2 >= len_side_1 and len_side_2 >= len_side_3:
                tmp_triangles.append([tmp_triangles[0][0], tmp_triangles[0][1],
                                      ((tmp_triangles[0][2][0] + tmp_triangles[0][1][0]) / 2,
                                       (tmp_triangles[0][2][1] + tmp_triangles[0][1][1]) / 2)])

                tmp_triangles.append([tmp_triangles[0][0],
                                      ((tmp_triangles[0][2][0] + tmp_triangles[0][1][0]) / 2,
                                       (tmp_triangles[0][2][1] + tmp_triangles[0][1][1]) / 2),
                                      tmp_triangles[0][2]])

                '''draw line'''
                line_coord = ([((tmp_triangles[0][2][0] + tmp_triangles[0][1][0]) / 2,
                                (tmp_triangles[0][2][1] + tmp_triangles[0][1][1]) / 2),
                               tmp_triangles[0][0]])

                line_x, line_y = zip(*line_coord)  # create lists of x and y values
                plt.plot(line_x, line_y, color='black', alpha=1)

                tmp_triangles.remove(tmp_triangles[0])


            elif len_side_3 > 1 and len_side_3 > len_side_1 and len_side_3 > len_side_2:
                tmp_triangles.append([tmp_triangles[0][0], tmp_triangles[0][1],
                                      ((tmp_triangles[0][2][0] + tmp_triangles[0][0][0]) / 2,
                                       (tmp_triangles[0][2][1] + tmp_triangles[0][0][1]) / 2)])

                tmp_triangles.append([((tmp_triangles[0][2][0] + tmp_triangles[0][0][0]) / 2,
                                       (tmp_triangles[0][2][1] + tmp_triangles[0][0][1]) / 2),
                                      tmp_triangles[0][1], tmp_triangles[0][2]])

                '''draw line'''
                line_coord = ([((tmp_triangles[0][2][0] + tmp_triangles[0][0][0]) / 2,
                                (tmp_triangles[0][2][1] + tmp_triangles[0][0][1]) / 2),
                               tmp_triangles[0][1]])

                line_x, line_y = zip(*line_coord)  # create lists of x and y values
                plt.plot(line_x, line_y, color='black', alpha=1)

                tmp_triangles.remove(tmp_triangles[0])



            elif len_side_1 <= 1 and len_side_2 <= 1 and len_side_3 <= 1:

                all_splited_triangles.append(tmp_triangles[0])
                tmp_triangles.remove(tmp_triangles[0])

    return all_splited_triangles


def vector_rotation(dot, axis_of_rotation, angle):
    '''Поворот точки dot (вектора) относительно точки axis_of_rotation на угол angle
    '''

    #         См. Википедия Матрица поворота
    #         x' = x*cosΘ + y*sinΘ
    #         y' = -x*sinΘ + y*cosΘ

    translated_x = dot[0] - axis_of_rotation[0]
    translated_y = dot[1] - axis_of_rotation[1]

    new_dot = np.array((translated_x * np.cos(np.radians(-angle))
                        + translated_y * np.sin(np.radians(-angle)),
                        -translated_x * np.sin(np.radians(-angle))
                        + translated_y * np.cos(np.radians(-angle)))) + np.array(axis_of_rotation)

    return new_dot


def triangle_area(triangle):
    '''Вычисление площади треугольника по теореме Герона
    '''
    len_side_1 = np.sqrt((triangle[0][0] - triangle[1][0]) ** 2 +
                         (triangle[0][1] - triangle[1][1]) ** 2)
    len_side_2 = np.sqrt((triangle[2][0] - triangle[1][0]) ** 2 +
                         (triangle[2][1] - triangle[1][1]) ** 2)
    len_side_3 = np.sqrt((triangle[0][0] - triangle[2][0]) ** 2 +
                         (triangle[0][1] - triangle[2][1]) ** 2)
    p = (len_side_1 + len_side_2 + len_side_3) / 2
    area = np.sqrt(p * (p - len_side_1) * (p - len_side_2) * (p - len_side_3))
    #     print(p, len_side_1, len_side_2, len_side_3)
    if np.isnan(area) == False:
        return area
    else:
        return 0


def MultipleOperator(triangles, partition_number):
    # Передаём массив треугольников, выбирается самый большой треугольник и делится на 2 рандомные части,
    # затем снова берется самый большой (или с самым плохим заполнением) и так partition_number-1 раз.
    # В ходе выполнения происходит вызов функции самой себя
    while partition_number != 1:
        # Выбираем треугольник с самой большой площадью
        area_list = []
        for j in range(len(triangles)):
            area_list.append(triangle_area(triangles[j]))

        # divisible_triangle
        triangle = triangles[area_list.index(np.max(area_list))]

        angle = random.randint(0, 2)
        random_value = random.uniform(0.2, 0.8)

        triangles.append([((triangle[(angle - 1) % 3][0] * random_value + triangle[angle][0] * (1 - random_value)),
                           (triangle[(angle - 1) % 3][1] * random_value + triangle[angle][1] * (1 - random_value))),
                          triangle[angle], triangle[(angle + 1) % 3]])
        triangles.append([triangle[(angle - 1) % 3],
                          ((triangle[(angle - 1) % 3][0] * random_value + triangle[angle][0] * (1 - random_value)),
                           (triangle[(angle - 1) % 3][1] * random_value + triangle[angle][1] * (1 - random_value))),
                          triangle[(angle + 1) % 3]])

        # Удаляем треугольник, который разбили на части
        triangles.remove(triangle)

        partition_number -= 1
    return triangles


def fillTriangle(triangle):
    '''Заполняет переданный треугольник треугольником из базиса

    В ходе заполнения перебираются все варианты треугольников с разными углами в основании
    Выдается треугольник с лучшим значением заполнения
    '''

    angle_0 = angle_finder(triangle[1], triangle[0], triangle[2])
    angle_1 = angle_finder(triangle[2], triangle[1], triangle[0])
    angle_2 = angle_finder(triangle[0], triangle[2], triangle[1])

    angles = [angle_0, angle_1, angle_2]

    possible_fill_coords = []
    possible_fill_areas = []

    for index in range(len(angles)):

        len_side_1 = np.sqrt((triangle[0][0] - triangle[1][0]) ** 2 +
                             (triangle[0][1] - triangle[1][1]) ** 2)
        len_side_2 = np.sqrt((triangle[2][0] - triangle[1][0]) ** 2 +
                             (triangle[2][1] - triangle[1][1]) ** 2)
        len_side_3 = np.sqrt((triangle[0][0] - triangle[2][0]) ** 2
                             + (triangle[0][1] - triangle[2][1]) ** 2)

        for k in range(len(basis_angles) - 1):
            if basis_angles[k] < angles[index] < basis_angles[k + 1]:
                fill_angle = basis_angles[k]

        for j in range(19):
            if np.linspace(0, 1, 20)[j] <= len_side_1 and np.linspace(0, 1, 20)[j + 1] > len_side_1:
                fill_side_1 = np.linspace(0, 1, 20)[j]
            elif np.linspace(0, 1, 20)[j + 1] == len_side_1:
                fill_side_1 = np.linspace(0, 1, 20)[j + 1]

        '''
        найдем точку пересечения вектора боковой стороны треугольника
        и луча, выходящего из 0 под углом fill_angle:
        '''
        line1 = line(triangle[(index + 1) % 3], triangle[(index - 1) % 3])
        # Для нахождения 2 точки в line2 (конца второго вектора) мы применяем матрицу поворота
        # к первому вектору

        line2 = line(triangle[index],
                     tuple(vector_rotation(triangle[(index - 1) % 3], triangle[index], fill_angle)))

        R = intersection(line1, line2)

        len_R = (((R[0] - triangle[index][0]) ** 2 + (R[1] - triangle[index][1]) ** 2) ** (1 / 2))

        for j in range(19):
            if np.linspace(0, 1, 20)[j] <= len_R and np.linspace(0, 1, 20)[j + 1] > len_R:
                fill_side_2 = np.linspace(0, 1, 20)[j]
            elif np.linspace(0, 1, 20)[j + 1] == len_R:
                fill_side_2 = np.linspace(0, 1, 20)[j + 1]

        fill_coord = [tuple(np.array(triangle[(index - 1) % 3]) * (fill_side_1 / len_side_1) +
                            np.array(triangle[index]) * (1 - (fill_side_1 / len_side_1))),
                      triangle[index],
                      tuple(np.array(R) * (fill_side_2 / len_R) +
                            np.array(triangle[index]) * (1 - (fill_side_2 / len_R)))]
        #             fill_coord = [all_triangles[i][0], all_triangles[i][1], all_triangles[i][2]]
        possible_fill_coords.append(fill_coord)
        #         print('fill_coord', fill_coord)
        #         print('triangle_area', triangle_area(fill_coord))
        if np.isnan(triangle_area(fill_coord)) == False:
            possible_fill_areas.append(triangle_area(fill_coord))
        else:
            possible_fill_areas.append(0)
    #         print('triangle_area', triangle_area(fill_coord))

    # print('possible_fill_coords', possible_fill_coords)
    # print('possible_fill_areas', possible_fill_areas)

    best_fill_coord = possible_fill_coords[possible_fill_areas.index(np.max(possible_fill_areas))]

    return best_fill_coord


def fillPolygon(all_triangles):
    '''Заполнение, полученных в ходе разбиения, треугольников со сторонами <= 1 и их рисовка

    '''
    polygon_area = 0
    polygon_filled_area = 0

    for i in range(len(all_triangles)):

        polygon_area += triangle_area(all_triangles[i])

        """нахождение заполненной площади"""

        variants_of_division = []
        all_areas = []

        #         print('all_triangles[i]', all_triangles[i])

        for partition_number in range(1, 4):
            filled_area = 0
            divided_triangles = MultipleOperator([all_triangles[i]], partition_number)

            #             print('divided_trianlges', divided_triangles)
            variants_of_division.append(divided_triangles)

            for triangle in divided_triangles:
                # print('triangle:', triangle)
                # print(triangle)

                filled_area += triangle_area(fillTriangle(triangle))
            #                 print('filled_area', filled_area)
            all_areas.append(filled_area)

        #         print(all_areas)
        #         print(variants_of_division)

        best_fill_variant = variants_of_division[all_areas.index(np.max(all_areas))]

        for j in range(len(best_fill_variant)):
            fill_coord = fillTriangle(best_fill_variant[j])

            # fill_coord=fillTriangle(all_triangles[i])

            polygon_filled_area += triangle_area(fill_coord)

            fill_coord.append(fill_coord[0])  # repeat the first point to create a 'closed loop'

            xs, ys = zip(*fill_coord)  # create lists of x and y values

            my_plot = plt.fill(xs, ys, alpha=0.7)
            '''аннотация треугольников'''
        #             plt.annotate('{}'.format(i),
        #                          xy=((all_triangles[i][2][0] +
        #                               all_triangles[i][1][0] + all_triangles[i][0][0])/3,
        #                              (all_triangles[i][2][1] +
        #                               all_triangles[i][1][1] + all_triangles[i][0][1])/3))

        #         plt.setp(my_plot, facecolor='red')

        '''
        конец куска заполнения
            '''
    print('filled_area/polygon_area', polygon_filled_area, '/', polygon_area,
          ':', polygon_filled_area / polygon_area * 100, '%')


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def angle_finder(coord_1, coord_2, coord_3):
    '''Нахождение угла между векторами. !!!ОТСЧЕТ ИДЕТ ПО ЧАСОВОЙ СТРЕЛКЕ!!!

    Перенос, чтобы точка соединения векторов была (0, 0)
    '''
    new_x_1 = coord_1[0] - coord_2[0]
    new_y_1 = coord_1[1] - coord_2[1]

    new_x_2 = coord_3[0] - coord_2[0]
    new_y_2 = coord_3[1] - coord_2[1]

    angle_1 = (angle_between([new_x_1, new_y_1], [1, 0]) * np.sign(new_y_1)) % 360
    angle_2 = (angle_between([new_x_2, new_y_2], [1, 0]) * np.sign(new_y_2)) % 360
    angle = (angle_1 - angle_2) % 360
    return angle


def all_angles_finder(coords):
    '''Нахождение набора углов многоугольника
    '''
    all_angles = []
    coords.insert(0, coords[-1])
    coords.append(coords[1])
    for i in range(1, len(coords) - 1):
        angle = angle_finder(coords[i - 1], coords[i], coords[i + 1])
        # print(i-1, angle)
        all_angles.append(angle)

    return all_angles


if __name__ == "__main__":

    number_of_dots = 14  # random.randint(3, 10)
    # random.seed(921)
    # np.random.seed(921)

    all_x, all_y = generate_dots(number_of_dots, 2)

    coords = [(all_x[i], all_y[i]) for i in range(len(all_x))]

    all_angles = all_angles_finder(coords)

    drawPolygon()
    divisible_polygons = splitPolygon()

    all_triangles = drawTriangle(divisible_polygons)

    basis, basis_angles = genBasis()

    all_splited_triangles = []
    all_splited_triangles = split_triangles_into_shorter_ones(all_triangles)

    fillPolygon(all_splited_triangles)

    plt.scatter(x=all_x, y=all_y, marker='o', c='r', edgecolor='b')
    ''' аннотация точек'''
    for i in range(len(all_x)):
        plt.annotate(i, (all_x[i], all_y[i]))
    plt.grid()
    plt.show()
# % reset_selective -f pyobject