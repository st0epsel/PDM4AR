from collections.abc import Sequence
from turtle import back

from dg_commons import SE2Transform
import numpy as np
import numpy.typing as npt

from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.utils import extract_path_points


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


class Dubins(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> list[SE2Transform]:
        """Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a list[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


class ReedsShepp(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a list[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


def add_tuples(a: tuple, b: tuple) -> tuple:
    return (a[0] + b[0], a[1] + b[1])


def diff_tuples(a: tuple, b: tuple) -> tuple:
    return (b[0] - a[0], b[1] - a[1])


def rotation_matrix_2d(theta: float) -> npt.NDArray[np.float128]:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float128)


def angled_unit_vector(theta: float) -> npt.NDArray[np.float128]:
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float128)


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    return DubinsParam(wheel_base / np.tan(max_steering_angle))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    if float(np.linalg.norm(vector, ord=2)) == 0.0:
        return np.zeros_like(vector)
    return vector / float(np.linalg.norm(vector, ord=2))


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    pos = current_config.p
    theta = current_config.theta

    L_centre = pos + angled_unit_vector(theta + np.pi / 2) * radius
    left_circle = Curve.create_circle(
        center=SE2Transform(p=L_centre.tolist(), theta=0),
        config_on_circle=current_config,
        radius=radius,
        curve_type=DubinsSegmentType.LEFT,
    )

    R_centre = pos + angled_unit_vector(theta - np.pi / 2) * radius
    right_circle = Curve.create_circle(
        center=SE2Transform(p=R_centre.tolist(), theta=0),
        config_on_circle=current_config,
        radius=radius,
        curve_type=DubinsSegmentType.RIGHT,
    )

    return TurningCircle(left=left_circle, right=right_circle)


def calc_arc_angle_obj(curve_obj: Curve) -> float:
    length = calc_arc_angle(curve_obj.start_config, curve_obj.end_config, curve_obj.center)
    return length


def calc_arc_angle(start_config: SE2Transform, end_config: SE2Transform, circle_center: SE2Transform) -> float:
    P1 = start_config.p
    P2 = end_config.p
    heading1 = start_config.theta
    heading2 = end_config.theta
    C = circle_center.p
    vec_C_to_P1 = P1 - C
    vec_C_to_P2 = P2 - C
    radius1 = float(np.linalg.norm(vec_C_to_P1))
    radius2 = float(np.linalg.norm(vec_C_to_P2))
    # print(f"Getting angular dist ... ")
    # print(f"   P1: {np.round(P1,3)} {heading1}, P2: {np.round(P2,3)} {heading2}, C: {np.round(C,3)}, radius: {radius1}")
    assert np.isclose(radius1, radius2, atol=1e-4), f"Points not on the same circle (R1 = {radius1}, R2 = {radius2})"
    assert radius1 > 1e-6, "Radius is zero"

    u_vec_C_to_P1 = normalize_vector(vec_C_to_P1)
    u_vec_C_to_P2 = normalize_vector(vec_C_to_P2)

    # Determine turning direction
    dir1 = np.cross(angled_unit_vector(heading1), u_vec_C_to_P1)
    dir2 = np.cross(angled_unit_vector(heading2), u_vec_C_to_P2)

    assert np.sign(dir1) == np.sign(
        dir2
    ), f"Distance between arc points cannot be found, directions oppose each other\n heading1: {heading1}, heading2: {heading2}, dir1: {dir1}, dir2: {dir2}"

    # Calculate arc angle between points
    theta1 = float(np.arctan2(vec_C_to_P1[1], vec_C_to_P1[0]))
    theta2 = float(np.arctan2(vec_C_to_P2[1], vec_C_to_P2[0]))
    arc_angle = theta2 - theta1

    if dir1 > 0:
        dist = mod_2_pi(-arc_angle)
    else:
        dist = mod_2_pi(arc_angle)
    # print(f"   distance: {dist}")
    return dist


def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> list[Line]:
    r1 = circle_start.radius
    r2 = circle_end.radius
    center1 = circle_start.center.p
    center2 = circle_end.center.p
    dir1 = circle_start.type
    dir2 = circle_end.type
    centerline_vector = np.subtract(center2, center1)
    centerline_length = float(np.sqrt(centerline_vector[0] ** 2 + centerline_vector[1] ** 2))

    # Circle contained within other circle
    if centerline_length < abs(r1 - r2) and r1 != r2:
        return []

    # For identical circles (Attention at directions)
    if centerline_length == 0 and np.isclose(r1, r2) and dir1 == dir2:
        return [Line(circle_start.start_config, circle_start.start_config)]

    centerline_dir = centerline_vector / centerline_length  # Unit centerline vector
    centerline_theta = float(np.arctan2(center2[1] - center1[1], center2[0] - center1[0]))

    # Circle contained within other circle, but 'kissing'
    #     Circle_end is bigger
    if centerline_length == abs(r2 - r1) and dir1 == dir2:
        inter_point = center1 - centerline_dir * r1
        heading = centerline_theta - int(dir1) * np.pi / 2
        intersection_line = Line(
            SE2Transform(inter_point.tolist(), heading), SE2Transform(inter_point.tolist(), heading)
        )
        return [intersection_line]

    if dir1 == dir2:
        # --- Calculate non-diagonal entries ---
        # For Dir1 == Dir2
        beta = np.arcsin((r1 - r2) / centerline_length)
        tangent_theta = mod_2_pi(centerline_theta - int(dir1) * beta)
        uV = angled_unit_vector(tangent_theta - int(dir1) * np.pi / 2)
        Q1 = center1 + r1 * uV
        Q2 = center2 + r2 * uV
        return [Line(SE2Transform(Q1.tolist(), tangent_theta), SE2Transform(Q2.tolist(), tangent_theta))]

    if dir1 != dir2:
        # Check if circles 'kiss'
        if centerline_length == r1 + r2:
            inter_point = center1 + centerline_dir * r1
            theta = centerline_theta + int(dir1) * np.pi / 2
            inter_point_obj = SE2Transform(inter_point.tolist(), theta)
            return [Line(inter_point_obj, inter_point_obj)]

        # --- Calculate diagonal entries ---
        elif centerline_length > r1 + r2:
            beta = np.arccos((r1 + r2) / centerline_length)
            theta = centerline_theta - int(dir1) * beta
            uV = angled_unit_vector(theta)
            Q1 = center1 + r1 * uV
            Q2 = center2 - r2 * uV
            return [
                Line(
                    SE2Transform(Q1.tolist(), theta + int(dir1) * np.pi / 2),
                    SE2Transform(Q2.tolist(), theta + int(dir1) * np.pi / 2),
                )
            ]
    return []


def find_equ_r_helper_circles(circle_start: Curve, circle_end: Curve) -> list[Curve]:
    r1 = circle_start.radius
    r2 = circle_end.radius
    center1 = circle_start.center.p
    center2 = circle_end.center.p
    dir1 = circle_start.type
    dir2 = circle_end.type
    assert dir1 == dir2, "Circles are of unequal type"
    dir3 = -dir1
    r3 = r1
    assert r1 == r2, "Circles are of unequal radius"
    centerline_vector = np.subtract(center2, center1)
    centerline_length = float(np.sqrt(centerline_vector[0] ** 2 + centerline_vector[1] ** 2))
    centerline_theta = float(np.arctan2(centerline_vector[1], centerline_vector[0]))

    # If no solution possible, return empty list
    if centerline_length > 4 * r1 or np.array_equal(center1, center2):
        return []

    output_curves = []

    for sign in [1, -1]:
        alpha = sign * np.arccos(centerline_length / (4 * r1))
        angle_leg1 = centerline_theta + alpha
        angle_leg2 = np.pi + centerline_theta - alpha
        dir_leg1 = angled_unit_vector(angle_leg1)
        dir_leg2 = angled_unit_vector(angle_leg2)
        center3 = center1 + dir_leg1 * 2 * r1
        center3_obj = SE2Transform(center3.tolist(), 0)
        contact_point1 = center1 + dir_leg1 * r1
        contact_point1_obj = SE2Transform(contact_point1.tolist(), angle_leg1 + dir1 * np.pi / 2)
        contact_point2 = center2 + dir_leg2 * r2
        contact_point2_obj = SE2Transform(contact_point2.tolist(), angle_leg2 + dir1 * np.pi / 2)
        """out_var = {
            "alpha": alpha,
            "angle_leg1": angle_leg1,
            "angle_leg2": angle_leg2,
            "center3": center3,
            "contact_point1": contact_point1,
            "contact_point2": contact_point2,
        }
        for var_name, var in out_var.items():
            print(f"{var_name}: {np.round(var,3)}")
        """
        helper_circle = Curve(
            start_config=contact_point1_obj,
            end_config=contact_point2_obj,
            center=center3_obj,
            radius=r1,
            curve_type=DubinsSegmentType(-dir1),
            arc_angle=calc_arc_angle(contact_point1_obj, contact_point2_obj, center3_obj),
        )
        output_curves.append(helper_circle)
    # print(f"center: {center3}, contact_point1: {contact_point1_obj}, contact_point2: {contact_point2_obj}")
    return output_curves


def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:

    # Calculate immediate start & end turning circles
    start_circles = calculate_turning_circles(start_config, radius)
    end_circles = calculate_turning_circles(end_config, radius)
    optimal_path_length = np.inf
    optimal_path = []
    print("\n")
    print(f"start_config: {start_config}, end_config: {end_config}")

    for start_circle in [start_circles.left, start_circles.right]:
        print(f"\nstart_circle: {start_circle}")
        for end_circle in [end_circles.left, end_circles.right]:
            print(f"\nend_circle: {end_circle}")

            # --- CSC ---
            print(f"\nCSC:")
            tangents = calculate_tangent_btw_circles(start_circle, end_circle)
            if tangents:
                tangent = tangents[0]

                print(f"  tangent: {tangent}")
                dubins_start_circle = Curve(
                    start_config=start_config,
                    end_config=tangent.start_config,
                    center=start_circle.center,
                    radius=start_circle.radius,
                    curve_type=start_circle.type,
                    arc_angle=calc_arc_angle(start_config, tangent.start_config, start_circle.center),
                )
                dubins_end_circle = Curve(
                    start_config=tangent.end_config,
                    end_config=end_config,
                    center=end_circle.center,
                    radius=end_circle.radius,
                    curve_type=end_circle.type,
                    arc_angle=calc_arc_angle(tangent.end_config, end_config, end_circle.center),
                )
                path_length = tangent.length + dubins_start_circle.length + dubins_end_circle.length
                if path_length < optimal_path_length:
                    optimal_path_length = path_length
                    optimal_path = [dubins_start_circle, tangent, dubins_end_circle]

            # --- CCC ---
            print(f"\nCCC")
            if start_circle.type == end_circle.type:
                helper_circles = find_equ_r_helper_circles(start_circle, end_circle)
                for helper_circle in helper_circles:
                    print(f"  helper_circle: {helper_circle}")
                    dubins_start_circle = Curve(
                        start_config=start_config,
                        end_config=helper_circle.start_config,
                        center=start_circle.center,
                        radius=start_circle.radius,
                        curve_type=start_circle.type,
                        arc_angle=calc_arc_angle(start_config, helper_circle.start_config, start_circle.center),
                    )
                    dubins_end_circle = Curve(
                        start_config=helper_circle.end_config,
                        end_config=end_config,
                        center=end_circle.center,
                        radius=end_circle.radius,
                        curve_type=end_circle.type,
                        arc_angle=calc_arc_angle(helper_circle.end_config, end_config, end_circle.center),
                    )
                    path_length = helper_circle.length + dubins_start_circle.length + dubins_end_circle.length
                    if path_length < optimal_path_length:
                        optimal_path_length = path_length
                        optimal_path = [dubins_start_circle, helper_circle, dubins_end_circle]

    return optimal_path


def compare_spline_to_dubins(
    start_config: SE2Transform, end_config: SE2Transform, radius: float
) -> tuple[float, float, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare the Dubins path and a cubic Hermite spline between two configurations.

    Returns:
        dubins_length: optimal Dubins path length
        spline_length: numerical length of the Hermite spline
        is_feasible: True if spline curvature ≤ 1 / radius everywhere
    """
    # TODO implement here your solution
    dubins_path = calculate_dubins_path(start_config=start_config, end_config=end_config, radius=radius)
    dubins_length = 0.0
    for section in dubins_path:
        dubins_length += section.length

    spline_length = 0.0

    p0 = np.zeros(2, dtype=float)
    p1 = np.zeros(2, dtype=float)
    t0 = np.zeros(2, dtype=float)
    t1 = np.zeros(2, dtype=float)

    t0[0] = np.cos(start_config.theta)
    t0[1] = np.sin(start_config.theta)
    t1[0] = np.cos(end_config.theta)
    t1[1] = np.sin(end_config.theta)

    p0 = start_config.p
    p1 = end_config.p

    distance = p1 - p0
    scale = np.linalg.norm(distance)

    t0 = t0 * scale
    t1 = t1 * scale

    is_feasible = True

    def hermite(s):
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1

    spline_length = 0.0  # Replace with actual spline length calculation
    is_feasible = True  # Replace with actual feasibility check

    steps = 100
    s_values = np.linspace(0.0, 1.0, steps + 1)
    points = np.array([hermite(s) for s in s_values])

    for i in range(steps):
        # Sum the length of all 100 segments
        segment_length = np.linalg.norm(points[i + 1] - points[i])
        spline_length += segment_length

    for i in range(1, steps):
        a = points[i - 1]
        b = points[i]
        c = points[i + 1]

        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ac = np.linalg.norm(c - a)
        if ab < 0.000001 or bc < 0.000001:
            continue

        cos_argument = np.clip((ac**2 - ab**2 - bc**2) / (-2 * ab * bc), -1, 1)
        angle = np.arccos(cos_argument)
        absolute_angle = np.pi - angle
        average_length = (ab + bc) / 2.0

        kappa = absolute_angle / average_length
        if kappa > (1 / radius):
            is_feasible = False
            break

    if start_config.p[0] == end_config.p[0] and start_config.p[1] == end_config.p[1]:
        if start_config.theta == end_config.theta:
            is_feasible = True
        else:
            is_feasible = False

    return dubins_length, spline_length, is_feasible, t0, t1, p0, p1


def reverse_config(config: SE2Transform) -> SE2Transform:
    # Flip car's heading by 180 degrees
    return SE2Transform(config.p.tolist(), mod_2_pi(config.theta + np.pi))


def reverse_segment(segment: Segment) -> Segment:
    new_gear = Gear(segment.gear.value * -1)

    if segment.type == DubinsSegmentType.STRAIGHT:
        return Line(
            start_config=reverse_config(segment.start_config),
            end_config=reverse_config(segment.end_config),
            gear=new_gear,
        )
    else:
        return Curve(
            start_config=reverse_config(segment.start_config),
            end_config=reverse_config(segment.end_config),
            center=segment.center,
            radius=segment.radius,
            curve_type=DubinsSegmentType(-1 * segment.type),
            arc_angle=segment.arc_angle,
            gear=new_gear,
        )


def reverse_path(path: Path) -> Path:
    reversed_path = []
    for segment in path:
        reversed_path.append(reverse_segment(segment))
    return reversed_path


def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    fwd_path = calculate_dubins_path(start_config=start_config, end_config=end_config, radius=radius)
    fwd_path_length = sum(segment.length for segment in fwd_path)

    backward_path_deriv = calculate_dubins_path(reverse_config(start_config), reverse_config(end_config), radius)
    backward_path_length = sum(segment.length for segment in backward_path_deriv)

    if fwd_path_length < backward_path_length:
        return fwd_path
    else:
        backward_path = reverse_path(backward_path_deriv)
        return backward_path


if __name__ == "__main__":
    center1 = SE2Transform((0, 0), 0)
    center2 = SE2Transform((8, 0), 0)
    on_circle1 = SE2Transform((0, 2), np.pi / 2)
    on_circle2 = SE2Transform((8, 2), np.pi / 2)
    curve1 = Curve.create_circle(center1, on_circle1, 2.0, DubinsSegmentType.LEFT)
    curve2 = Curve.create_circle(center2, on_circle2, 2.0, DubinsSegmentType.LEFT)
    tangent = calculate_tangent_btw_circles(curve1, curve2)[0]
    print(f"tangent: {tangent}\ntangent.start_config: {tangent.start_config}, tangent end config: {tangent.end_config}")
    print(calculate_dubins_path(SE2Transform([0.0, 0.0], 0), SE2Transform([0.0, 7.0], 0), 3.5))
    """
    turning_circles = calculate_turning_circles(center1, 2.0)
    for circle in [turning_circles.left, turning_circles.right]:
        print(f"circle1: {circle}\n center: {circle.center}, radius: {circle.radius}, direction: {circle.type}")
    """
