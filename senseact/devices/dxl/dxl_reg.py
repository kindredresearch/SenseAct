"""Register block."""

#pylint: disable=too-many-arguments,invalid-name,unused-argument
#pylint: disable=missing-docstring
#pylint: disable=too-many-return-statements

import array
import numpy as np
from .dxl_unit_conv import UnitConversion


class Reg(object):
    """A class representing Dynamixel Register.

    A register is a logical, not physical register.

    Attributes:
        name: A string (semantics of the register)
        width: An integer representing number of bytes (1 byte or 2)
        x0: An initial factory-default value (byte or byte pair)
        x_lim: A tuple of lower and upper limits on the integer value
            a register can take
        unit: An optional unit_conversion that translates integer sensor
            readings into standard physical quantities (angles, forces, etc.)
        offset: An integer representing the address of the register in the DXL control table

    """

    def __init__(self, name, x0_bytes,
                 x_lim=None,
                 unit=UnitConversion(),
                 offset=None,
                 w=None):
        """Inits Reg objects with device specific parameters.

        Args:
            name: A string The label of the register (e.g. "present_speed")
            x0_bytes: An int or an int pair factory default value
            x_lim: None or (low, high) tuple lower and upper (inclusive)
                sensor limits
            unit: An instance of unit_conv.UnitConversion
            w: A boolean, True iff writeable register
            offset: An integer. The address of the register in the DXL control table

        """
        self.name = name
        if isinstance(x0_bytes, int):
            self.width = 1
            self.x0 = x0_bytes
            self.x0_bytes = (x0_bytes,)
        else:
            self.width = len(x0_bytes)
            self.x0 = 0
            for ii, vv in enumerate(x0_bytes):
                self.x0 += vv << (ii * 8)
            self.x0_bytes = x0_bytes
        self.x_lim = x_lim
        if isinstance(x_lim, tuple):
            low, high = x_lim
            assert low <= self.x0 <= high
        self.unit = unit
        if unit:
            unit.set_x_lim(x_lim)
        self.offset = offset

    @property
    def dtype(self):
        return np.dtype(self.dtype_string)

    @property
    def dtype_string(self):
        return '<u%i' % self.width

    @property
    def y_min(self):
        return self.unit.y_min

    @property
    def y_max(self):
        return self.unit.y_max

    @property
    def y0(self):
        return self.unit.fwd(self.x0)


class ContiguousRegisters(object):
    """Syntactic support class for accessing the Register objects.
    
    The Register objects make up the FactoryDefaults singleton.
    
    Attributes:
        ret_dxl_type: A bool flag. True for ctypes driver and False for pyserial driver. This flag is
                      used during data conversion (regular units to dxl units and vice versa).
        offset: An integer. The address of the register in the DXL control table
        width: An integer. The number of bytes to read from the DXL control table
    """

    def __init__(self, *regs, ret_dxl_type = False):
        """Inits class objects with device specific params.

        Args:
            *regs: Instances of dxl_reg.Reg for registers in the DXL control table
            ret_dxl_type: A bool flag. True for ctypes driver and False for pyserial driver. This flag is
                          used during data conversion (regular units to dxl units and vice versa).
        """
        self._regs = regs
        self.ret_dxl_type = ret_dxl_type

        no_offsets = True
        all_offsets = True
        for reg in self._regs:
            no_offsets = no_offsets and reg.offset is None
            all_offsets = all_offsets and reg.offset is not None
        if no_offsets and all_offsets:
            raise NotImplementedError('Empty register seq')
        elif no_offsets:
            self._calculate_and_assign_offsets()
        elif all_offsets:
            pass
        else:
            raise NotImplementedError('Some-but-not-all offsets assigned')
        self._assert_contiguous()
        self.offset = self._regs[0].offset
        self.width = sum(r.width for r in self._regs)

    def _pretty_version(self):
        """Description: Find model number using the first two bytes of the DXL control table"""
        if self._regs[0].offset == 0:
            if self._regs[0].x0 == 12:
                return 'AX12'
            elif self._regs[0].x0 == 18:
                return 'AX18'
            elif self._regs[0].x0 == 29:
                return 'MX28'
            elif self._regs[0].x0 == 54:
                return 'MX64'
            elif self._regs[0].x0 == 64:
                return 'MX106'
            return self._regs[0].x0_bytes
        else:
            return None

    def __str__(self):
        version = self._pretty_version()
        if version:
            return 'ContiguousRegisters{%s}' % version
        else:
            return NotImplemented

    def _calculate_and_assign_offsets(self, offset=0):
        for reg in self._regs:
            reg.offset = offset
            offset += reg.width

    def _assert_contiguous(self):
        offset = self._regs[0].offset
        for reg in self._regs:
            assert reg.offset == offset
            offset += reg.width

    def __iter__(self):
        return iter(self._regs)

    def len(self):
        return self.__len__()

    def __len__(self):
        return len(self._regs)

    def __contains__(self, key):
        if isinstance(key, str):
            for reg in self._regs:
                if reg.name == key:
                    return True
            else:
                return False
        elif isinstance(key, int):
            if key < 0:
                return False
            try:
                self._regs[key]
                return True
            except IndexError:
                return False
        elif isinstance(key, (list, tuple)):
            rval = self.subblock(key[0], key[-1])
            if list(key) != [r.name for r in rval]:
                return False
            return True
        return False

    def __getitem__(self, key):
        if isinstance(key, str):
            for reg in self._regs:
                if reg.name == key:
                    return reg
            else:
                raise KeyError(key)
        elif isinstance(key, int):
            return self._regs[key]
        elif isinstance(key, (list, tuple)):
            rval = self.subblock(key[0], key[-1])
            if list(key) != [r.name for r in rval]:
                raise KeyError()
            return rval
        raise TypeError(key)

    def subblock(self, first, last, ret_dxl_type=False):
        """ The DXL control table has multiple registers containing data regarding the current status and operation.
        Since the DXL uses asynchronous serial communication, we can only read chunks of data and not selective
        registers which are not adjoined. This function is used to select a particular chunk of registers to write to
        or read from.

        Args:
            first: A string The label of the register (e.g. "goal_pos")
            last: A string The label of the register (e.g. "present_speed")
                  These strings are listed in dxl_mx64.py and dxl_ax12.py for MX-64AT and AX-12 servos respectively.
            ret_dxl_type: A bool flag. True for ctypes driver and False for pyserial driver. This flag is
                          used during data conversion (regular units to dxl units and vice versa).

        Returns:
            An instance of ContiguousRegisters containing all the target registers

        """
        # N.B. possibly first == last
        regs = []
        for reg in self._regs:
            if regs or reg.name == first:
                regs.append(reg)
            if reg.name == last:
                break
        else:
            raise KeyError("{} is not a valid register name".format(last))
        if not regs:
            raise KeyError("{} is not a valid register name".format(first))
        if regs[-1].name == last:
            return ContiguousRegisters(*regs, ret_dxl_type=ret_dxl_type)
        else:
            raise KeyError(last)

    def vals_from_data(self, data):
        """Description: parse raw bytes and unit conversion

        Args:
            data: If ret_dxl_type = True: A list containing integer vals read from the DXL control table
                  If ret_dxl_type = False: Bytearray read from the DXL control table

        Returns:
            A list containing the current status and sensor readings from the DXL in regular format
        """
        if self.ret_dxl_type:
            return self.vals_from_dxl_data(data)
        
        raw_offset = 0
        vals = []
        for reg in self._regs:
            if reg.width == 1:
                b = data[raw_offset]
                raw_offset += 1
                vals.append(reg.unit.fwd(b))
            elif reg.width == 2:
                bl = data[raw_offset]
                br = data[raw_offset + 1]
                raw_offset += 2
                vals.append(reg.unit.fwd(bl + (br << 8)))
            else:
                raise NotImplementedError()
        return vals

    def vals_from_dxl_data(self, data):
        """Parse data in dxl format and unit conversion

        Args:
            data: A list containing integer vals read from the DXL control table

        Returns:
            A list containing the current status and sensor readings from the DXL in regular format
        """
        vals = []
        for ind, reg in enumerate(self._regs):
            b = data[ind]
            vals.append(reg.unit.fwd(b))
        return vals

    def data_from_vals(self, std_vals):
        """ Parse data in regular format and unit conversion to DXL format

        Args:
            std_vals: A list containing int and float vals to be converted to DXL format

        Returns:
            If ret_dxl_type = True: A list containing sensor values in DXL format
            If ret_dxl_type = False: A byte-array containing sensor values in DXL format
        """
        if self.ret_dxl_type:
            return self.dxl_data_from_vals(std_vals)

        data = array.array('B', [])
        if len(self._regs) != len(std_vals):
            raise ValueError()
        for std_val, reg in zip(std_vals, self._regs):
            int_val = reg.unit.inv(std_val)
            if reg.x_lim:
                x_min, x_max = reg.x_lim
                assert x_min <= int_val <= x_max, (
                    reg.name, std_val, x_min, int_val, x_max)
            if reg.width == 1:
                if 0 <= int_val < 256:
                    data.append(int_val)
                else:
                    raise ValueError(int_val)
            elif reg.width == 2:
                if 0 <= int_val < (1 << 16):
                    data.extend((int_val & 0xff, (int_val & 0xff00) >> 8))
                else:
                    raise ValueError(int_val)
            else:
                raise NotImplementedError()
        return data

    def dxl_data_from_vals(self, std_vals):
        """ Parse data in regular format and unit conversion to DXL format

        Args:
            std_vals: A list containing int and float vals to be converted to DXL format

        Returns:
            A list containing sensor values in DXL format
        """
        data = []
        if len(self._regs) != len(std_vals):
            raise ValueError()
        for std_val, reg in zip(std_vals, self._regs):
            int_val = reg.unit.inv(std_val)
            data.append(int_val)
        return data
