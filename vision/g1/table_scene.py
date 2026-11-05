"""Build a table-tennis table in a tag-centered Viser world frame."""


def add_table(scene, spec):
    width = float(spec["width_m"])
    length = float(spec["length_m"])
    height = float(spec["height_m"])
    thickness = float(spec["thickness_m"])
    inset = float(spec["tag_center_inset_m"])
    net_height = float(spec["net_height_m"])
    near_edge = -inset
    center_y = near_edge + length / 2
    far_edge = near_edge + length

    root = scene.add_frame("/table", show_axes=False)
    scene.add_box("/table/top", color=(23, 57, 86), dimensions=(width, length, thickness),
                  position=(0, center_y, -thickness / 2))
    for name, y in (("near_end", near_edge + 0.01), ("far_end", far_edge - 0.01)):
        scene.add_box(f"/table/lines/{name}", color=(245, 245, 240),
                      dimensions=(width, 0.02, 0.002), position=(0, y, 0.002))
    for name, x in (("left_side", -width / 2 + 0.01), ("right_side", width / 2 - 0.01)):
        scene.add_box(f"/table/lines/{name}", color=(245, 245, 240),
                      dimensions=(0.02, length, 0.002), position=(x, center_y, 0.002))
    scene.add_box("/table/lines/center", color=(235, 235, 230),
                  dimensions=(0.003, length, 0.002), position=(0, center_y, 0.002))

    scene.add_box("/table/net/fabric", color=(195, 200, 205), opacity=0.5,
                  dimensions=(width + 0.2, 0.004, net_height),
                  position=(0, center_y, net_height / 2))
    scene.add_box("/table/net/top_tape", color=(245, 245, 245),
                  dimensions=(width + 0.2, 0.008, 0.008), position=(0, center_y, net_height))
    for name, x in (("left", -width / 2 - 0.1), ("right", width / 2 + 0.1)):
        scene.add_box(f"/table/net/{name}_post", color=(65, 70, 75),
                      dimensions=(0.022, 0.022, net_height),
                      position=(x, center_y, net_height / 2))

    leg_height = height - thickness
    for x_name, x in (("left", -width / 2 + 0.2), ("right", width / 2 - 0.2)):
        for y_name, y in (("near", near_edge + 0.3), ("far", far_edge - 0.3)):
            scene.add_box(f"/table/legs/{x_name}_{y_name}", color=(65, 70, 75),
                          dimensions=(0.05, 0.05, leg_height),
                          position=(x, y, -(height + thickness) / 2))

    scene.add_label("/table/near_edge_label", "Near edge",
                    position=(-width / 2 - 0.18, near_edge, 0.04))
    return root, near_edge
