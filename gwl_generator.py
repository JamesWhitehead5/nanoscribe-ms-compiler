from enum import Enum
import os
import numpy as np
import csv
import random
import string
import math
import shutil

from gwl_generator import *

class TrimStyle(Enum):
    CIRCLE = 0
    SQUARE = 1
    MAX_CIRCLE = 2
    MAX_SQUARE = 3
    MAX_FIELD = 4


class ScattererStyle(Enum):
    CYLINDER = 0
    SPHERE = 1
    CONE = 2
    SQUARE_PRISM = 3


class Scatterer:
    def __init__(self, style, **kwargs):
        self.style = style
        if style == ScattererStyle.CYLINDER:
            self.diameter = kwargs['diameter']
            self.thickness = kwargs['thickness']
        elif style == ScattererStyle.SPHERE:
            self.diameter = kwargs['diameter']
        # elif style == scatter_style.CONE:
        #    self.diameter = kwargs['diameter']
        #    self.height = kwargs['height']
        # elif style == scatter_style.SQUARE_PRISM:
        #    self.side_length = kwargs['side_length']
        #    self.thickness = kwargs['thickness']
        else:
            raise ValueError("Unknown scatter type")

    def get_slice(self, x_offset, y_offset, z):
        if self.style == ScattererStyle.CYLINDER:
            if z > self.thickness:
                return ""
            else:
                return GwlCircle(
                    diameter=self.diameter
                ).to_gwl_filled(x_offset=x_offset, y_offset=y_offset, z=z)
        elif self.style == ScattererStyle.SPHERE:
            if z > self.diameter:
                return ""
            else:
                p = abs(self.diameter / 2 - z)
                circle_diameter = np.sqrt((self.diameter / 2) ** 2 - p ** 2)
                return GwlCircle(
                    diameter=circle_diameter
                ).to_gwl(x_offset=x_offset, y_offset=y_offset, z=z)


class Stage:
    piezo_range = 300.0

    @staticmethod
    def _goto_piezo_z(z):
        return "\nPiezoGotoZ {}\n".format(z)

    @staticmethod
    def goto_piezo_z(z):
        return Stage._goto_piezo_z(z) + "Wait 0.1\n"

    @staticmethod
    def _goto_piezo(x,y,z):
        return "\nPiezoGotoX {}\nPiezoGotoY {}\nPiezoGotoZ {}\nWait 0.1\n".format(x, y, z)

    @staticmethod
    def goto_piezo(x, y, z):
        in_bounds = Stage.piezo_coordinates_in_bounds(x,y,z)
        assert in_bounds, "Coordinates out of bounds"
        return Stage._goto_piezo(x, y, z)

    #to reduce slop in stage writing, it is advised that the stage approach the desired position from the same
    #direction each time.
    @staticmethod
    def piezo_approach_point(x, y, z):
        approach_distance = 1.0

        x0 = x + approach_distance
        y0 = y + approach_distance
        z0 = z + approach_distance

        assert Stage.piezo_coordinates_in_bounds(x0, y0, z0), "Warning! Stage does not have enough range to approach position properly"
        out_string = Stage._goto_piezo(x0, y0, z0)
        out_string += Stage.goto_piezo(x, y, z)

        return out_string

    @staticmethod
    def piezo_coordinates_in_bounds(x, y, z):
        return Stage.piezo_coordinate_in_bounds(x) and \
               Stage.piezo_coordinate_in_bounds(y) and \
               Stage.piezo_coordinate_in_bounds(z)

    @staticmethod
    def piezo_coordinate_in_bounds(coor):
        return 0.0 <= coor <= Stage.piezo_range




class Field:

    def __init__(self, x_origin, y_origin, x_span, y_span, name):
        # type: (float, float, float, float) -> object
        self.x_origin = x_origin
        self.y_origin = y_origin
        self.x_span = x_span
        self.y_span = y_span

        self.name = name

        self.scatterer_list = []
        self.x_list = []
        self.y_list = []

        #N = 10  # name length
        #self.name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

    # define x and y in absolute coordinites
    def add(self, scatterer, x, y):
        """

        :rtype: object
        """
        self.scatterer_list.append(scatterer)
        self.x_list.append(x - self.x_origin)
        self.y_list.append(y - self.y_origin)

    def is_point_inside(self, x, y):
        within_x = x >= self.x_origin - self.x_span/2.0 and x < self.x_origin + self.x_span / 2.0
        within_y = y >= self.y_origin - self.y_span/2.0 and y < self.y_origin + self.y_span / 2.0
        return within_x and within_y

    def to_gwl(self, z_slices):
        gwl_string = ""
        i = 0
        n = len(z_slices)
        for z in z_slices:
            i += 1
            print("Slicing layer {}/{}\n".format(i, n))
            for scatterer, x, y in zip(self.scatterer_list, self.x_list, self.y_list):
                buff = scatterer.get_slice(x_offset=x, y_offset=y, z=z)
                if buff != "":
                    gwl_string += buff + "Write\n"
        return gwl_string


class GwlMS:
    """
    :param filename: Filename
    :type filename: String

    :param piezo_range: Range that the piezo stage can move in um
    :type piezo_range: Float

    :param galvo_range: Range of galvo motion in um
    :type galvo_range: Float
    """

    # loads the first column of data in a text file
    @staticmethod
    def _readfile(filename):
        # print(os.getcwd())
        data = []
        with open(filename) as csvfile:
            fileReader = csv.reader(csvfile, delimiter='\t')
            for row in fileReader:
                data.append(float(row[0]))
        return data

    @staticmethod
    def _load_data_from_file(filename, scatterer_style):
        # dir_path = os.getcwd()
        # print("looking is folder: {}".format(dir_path))
        path = './'
        x = GwlMS._readfile(path + 'x' + filename)
        y = GwlMS._readfile(path + 'y' + filename)

        d = GwlMS._readfile(path + 'd' + filename)
        h = GwlMS._readfile(path + 'h' + filename)

        # save point to fields




        # recenter the MS
        #x -= (x_min / 2. + x_max / 2.)
        #y -= (y_min / 2. + y_max / 2.)
        assert len(x) == len(y) == len(d) == len(h), "Vectors in coordinates must be the same length"

        n_points = len(x)

        scatterers = []
        for i in range(n_points):
            cylinder_kargs = {'diameter': d[i], 'thickness': h[i]}
            scatterer_style = ScattererStyle.CYLINDER
            # if the fields are odd arranged
            scatterers.append(Scatterer(scatterer_style, **cylinder_kargs))

        print("Loaded {} points from file".format(n_points))

        return x, y, scatterers


    #field generation assume structure has no large open spacing so square packing of fields is optimal
    def _generate_fields(self, x, y, scatterers):
        assert len(x) == len(y) == len(scatterers)
        self.n_points = len(x)

        self.x_size = max(x) - min(x)
        self.y_size = max(y) - min(y)

        x_centroid = max(x) / 2.0 + min(x) / 2.0
        y_centroid = max(y) / 2.0 + min(y) / 2.0

        field_size = galvo_range
        field_x_n = int(self.x_size / field_size + 1)
        field_y_n = int(self.y_size / field_size + 1)

        # split points into fields
        self.fields = []
        # create a list of fields
        for i in range(field_x_n):
            for j in range(field_y_n):
                if field_x_n % 2 == 0:
                    field_origin_x = field_size * (i - field_x_n / 2.0 + 0.5)
                else:
                    field_origin_x = field_size * (i - field_x_n / 2.0 + 0.5)
                if field_y_n % 2 == 0:
                    field_origin_y = field_size * (j - field_y_n / 2.0 + 0.5)
                else:
                    field_origin_y = field_size * (j - field_y_n / 2.0 + 0.5)

                if abs(field_origin_x) < piezo_range / 2.0 and abs(field_origin_x) < piezo_range / 2.0:
                    print("Geometries outside writeable area")

                self.fields.append(
                    Field(
                        field_origin_x + x_centroid,
                        field_origin_y + y_centroid,
                        x_span=field_size,
                        y_span=field_size,
                        name=str((i, j))
                    )
                )

                print("Field ({}, {}) created with origin ({}, {})".format(i, j, field_origin_x, field_origin_y))



        # ms_coverage_x = field_x_n * field_size
        # ms_coverage_y = field_y_n * field_size

        n = 0

        # arrange points into fields
        for i in range(self.n_points):
            # This method is inefficient but much easier to program and understand
            for field in self.fields:
                if field.is_point_inside(x[i], y[i]):
                    #assert type(scatterers[i]) != type([])
                    #print("x: {} y: {}".format(x[i], y[i]))
                    #print("Width: {} Height: {}".format(scatterers[i].diameter, scatterers[i].thickness))
                    field.add(scatterer=scatterers[i], x=x[i], y=y[i])
                    n += 1
        print("Number of elements: {}".format(n))

    def __init__(self, filename, piezo_range, galvo_range, scatterer_style=ScattererStyle.CYLINDER):
        x, y, scatterers = GwlMS._load_data_from_file(filename, scatterer_style)

        self.filename = filename
        self.scatterer_style = scatterer_style

        self.piezo_range = piezo_range
        self.galvo_range = galvo_range
        self._generate_fields(x, y, scatterers)

    def get_point_lists(self):
        x_points = []
        y_points = []
        scatterers = []

        for field in self.fields:
            x_points += field.x_list
            y_points += field.y_list
            scatterers += field.scatterer_list
        return x_points, y_points, scatterers

    def write_gwl(self, z_slices, name):
        """
        :returns
        """

        # Generate GWL file for arranging the sliced scatterers
        with open(name + "_main.gwl", "w") as gwl_file:
            print(os.getcwd())
            with open("./GWL_HEADER.txt", "r") as gwl_header:
                for header_line in gwl_header:
                    gwl_file.write(header_line)

            field_counter = 0
            for field in self.fields:
                field_counter += 1
                print("Field {}/{}".format(field_counter, len(self.fields)))

                field_name = name + "_field_" + str(field_counter) + '.gwl'

                # append references to main gwl
                gwl_file.write(
                    Stage.piezo_approach_point(x=field.x_origin + piezo_range/2.0,
                                     y=field.y_origin + piezo_range/2.0,
                                     z=0.0)
                )
                gwl_file.write("\nFindInterfaceAt $interfacePos")
                gwl_file.write("\ninclude {}".format(field_name))

                # generate field files
                with open(field_name, "w") as field_file:
                    field_file.write(Stage.goto_piezo_z(0.0))
                    field_file.write(field.to_gwl(z_slices))

    def trim(self, trim_style, size=None):

        def trim_data(keep_condition):
            x_list_new = []
            y_list_new = []
            scatterer_list_new = []

            for field in self.fields:
                for x, y, scatterer in zip(field.x_list ,
                                           field.y_list,
                                           field.scatterer_list):
                    if keep_condition(x + field.x_origin, y + field.y_origin):
                        x_list_new.append(x + field.x_origin)
                        y_list_new.append(y + field.y_origin)
                        scatterer_list_new.append(scatterer)
            self._generate_fields(x=x_list_new, y=y_list_new, scatterers=scatterer_list_new)

        if trim_style == TrimStyle.CIRCLE:
            keep_condition = lambda x, y: math.sqrt(x ** 2 + y ** 2) < size / 2.0
            trim_data(keep_condition)
        elif trim_style == TrimStyle.SQUARE:
            keep_condition = lambda x, y: abs(x) < size / 2.0 and abs(y) < size / 2.0
            trim_data(keep_condition)
        elif trim_style == TrimStyle.MAX_CIRCLE:
            self.trim(trim_style=TrimStyle.CIRCLE, size=min([self.x_size, self.y_size]))
        elif trim_style == TrimStyle.MAX_SQUARE:
            self.trim(trim_style=TrimStyle.SQUARE, size=min([self.x_size, self.y_size]))
        elif trim_style == TrimStyle.MAX_FIELD:
            keep_condition = lambda x, y: \
                abs(x) < piezo_range / 2.0 and abs(y) < piezo_range / 2.0 or \
                abs(x) < piezo_range / 2.0 + galvo_range / 2.0 and abs(y) < piezo_range / 2.0 or \
                abs(y) < piezo_range / 2.0 + galvo_range / 2.0 and abs(x) < piezo_range / 2.0 or \
                math.sqrt((abs(x) - piezo_range / 2.0) ** 2 + (abs(y) - piezo_range / 2.0) ** 2) < galvo_range / 2.0
            trim_data(keep_condition)
        else:
            raise Exception('trim_style unknown')


class GwlCircle:
    def __init__(self, diameter, INTERPOINT_DISTANCE = 0.15):
        self.diameter = diameter

        #design number of points so that arc length between neighbooring points is constant

        circumference = math.pi*diameter
        n_points = int(circumference/INTERPOINT_DISTANCE) + 1

        self.t = 1.0 * np.arange(0, n_points) / n_points * 2 * np.pi
        self.x = diameter / 2.0 * np.sin(self.t)
        self.y = diameter / 2.0 * np.cos(self.t)

    def to_gwl(self, x_offset, y_offset, z):

        write_string = "%Writing circle at: ({}, {})\n".format(x_offset, y_offset)
        for i in range(len(self.t)):
            x_abs = x_offset + self.x[i]
            y_abs = y_offset + self.y[i]
            #using only two decimal places to save space. Also, resolution of galvo is much lower.
            write_string += GwlCircle._write_point_string(x_abs, y_abs, z)
        return write_string

    def to_gwl_filled(self, x_offset, y_offset, z, fill_point_spacing=0.3):
        if self.diameter < fill_point_spacing:
            return GwlCircle._write_point_string(x_offset, y_offset, z)
        else:
            next_radius = self.diameter - 2.0*fill_point_spacing
            if next_radius < 0: next_radius = 0.0
            next_circle = GwlCircle(next_radius)
            return self.to_gwl(x_offset, y_offset, z) + \
                   next_circle.to_gwl_filled(x_offset, y_offset, z, fill_point_spacing)

    @staticmethod
    def _write_point_string(x,y,z):
        return "{:.2f}\t{:.2f}\t{:.3f}\n".format(x,y,z)

if __name__ == "__main__":
    piezo_range = Stage.piezo_range
    galvo_range = 130.0 #is 135um but is reduced because features have finite size and ceneters can extend outside

    ms1 = GwlMS(filename="_500.0_1.5_150.0.out", piezo_range=piezo_range, galvo_range=galvo_range)
    ms2 = GwlMS(filename="_600.0_1.5_150.0.out", piezo_range=piezo_range, galvo_range=galvo_range)

    # ms1 = GwlMS(filename="_test_data.out", piezo_range=piezo_range, galvo_range=galvo_range)

    nominal_size = 300.0
    ms1.trim(trim_style=TrimStyle.CIRCLE, size=5.0/6*nominal_size)
    ms2.trim(trim_style=TrimStyle.CIRCLE, size=nominal_size)

    z_slices = np.arange(0.0, 6.0, 0.3)  # max height is 6.0um
    print("Writing structure to files...")
    ms1.write_gwl(z_slices, name="MS1")
    ms2.write_gwl(z_slices, name="MS2")
    print("Fin!")