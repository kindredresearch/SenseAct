# Background
>The DYNAMIXEL is a smart actuator system developed to be the exclusive connecting joints on a
robot or mechanical structure. DYNAMIXELS’ are designed to be modular and daisy chained on any robot

Dynamixel motors can be easily used to build custom robots ranging from 3-DOF robot arms to humanoids.
To control Dynamixel, communication should be established according to its protocol. We used DXL MX-64AT
and AX-12A servos for all our experiments. They use [Protocol 1.0](http://support.robotis.com/en/product/actuator/dynamixel/dxl_communication.htm)
for half duplex UART serial communication. The motors are driven by sending and receiving binary data.

We used a Xevelabs USB2AX v3.2a USB to TTL Dynamixel servo interface for connecting the motors to the computer.

# Setting the Baudrate and Device ID
The easiest way to configure a DXL device is via the [Dynamixel Wizard](http://support.robotis.com/en/software/roboplus/dynamixel_monitor.htm).
Unfortunately, the software is supported only on Windows. That said, we could also configure the device
using our custom python script:

- [dxl_basic_functions.py](dxl_basic_functions.py) can be used to configure DXL MX-64 series servos

Typically, new servos have default `device ID = 1` and `baudrate = 57142`. In that case, specify `baudrate = 57142` and `idn = 1` in lines 17 and 18 of the dxl_basic_functions script, respectively. For example, the following commands
to change the baudrate and device ID number to 1 million and 9 respectively.

```python
write_to_register('baud', 1000000)
write_to_register('bus_id', 9)

```

 We also recommend setting the [Return Delay Time](http://support.robotis.com/en/product/actuator/dynamixel/mx_series/mx-64at_ar.htm#Actuator_Address_05)
 to zero. That can be done using the following command:

 ```python
write_to_register('rtd', 0)
```

# Setting up the Dynamixel device for your first experiment
Identify the port and make it available to be used. If you are using a USB2AX device:

```
 sudo chmod a+rw /dev/ttyACM0
```

If you are using a Robotis USB2Dynamixel device:

```
 sudo chmod a+rw /dev/ttyUSB0
```

There are plenty of examples specified within the `senseact/lib/DynamixelSDK/python/protocol1_0/` folder.

Executing a chmod command to access the serial device every time we connect a serial device is annoying
to say the least. In order to allow a non-default user to use serial device, all we need to do is add the
user to the `dialout` group:

```bash
sudo adduser user_name dialout
```

However, for this change to take effect, the user has to logout and then login again.

# Uninstall Dynamixel library files
- `cd senseact/lib/DynamixelSDK/c/build/` 
- Depending on the OS and format, choose the right subfolder. For example, for a 64-bit linux platform, `cd linux64` 
- `sudo make uninstall`
- `cd ~/SenseAct` && `rm -rf senseact/lib/DynamixelSDK`

If the `lib/DynamixelSDK` directory was removed before `sudo make uninstall`, follow the instructions listed below. The setup script copies the library file to the root directory to handle the serial post. Hence, we need to remove some files from `/usr/local`:

 ```bash
 # Only needed if "lib/DynamixelSDK" directory was removed before "sudo make uninstall"
 rm /usr/local/lib_dxl_x64_c.so
 rm /usr/local/lib_dxl_x64_c.so.2
 rm /usr/local/lib_dxl_x64_c.so.2.0
 rm /usr/local/lib_dxl_x64_c.so.2.0.0
 rm /usr/local/include/dynamixel_sdk.h
 ```

# Additional resources
Please note that this is not meant to be an exhaustive resource. We found the following links to be useful for
additional information. If you need help with DXL troubleshooting, kindly reach out to use, we'd be happy to
help you.

- [DXL MX-64 docs](http://support.robotis.com/en/product/actuator/dynamixel/mx_series/mx-64at_ar.htm)
- [DXL AX-12 docs](http://support.robotis.com/en/product/actuator/dynamixel/ax_series/dxl_ax_actuator.htm)
- [Pypot](https://github.com/poppy-project/pypot)
- [Robotis DynamixelSDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)

# Troubleshooting Information

## ValueError: 0
```python
  from ._conv import register_converters as _register_converters
Succeeded to open the port!
Baudrate set to: 1000000
Timeout set to: 500!
b'[TxRxResult] There is no status packet!'
b'[TxRxResult] There is no status packet!'
Process DXLCommunicator-1:
Traceback (most recent call last):
  File "/home/gautham/.conda/envs/rllab3/lib/python3.5/multiprocessing/process.py", line 252, in _bootstrap
    self.run()
  File "/home/gautham/src/senseact/senseact/devices/dxl/dxl_communicator.py", line 101, in run
    vals = self.dxl_driver.read_a_block(self.port, self.idn, read_block, self.read_wait_time)
  File "/home/gautham/src/senseact/senseact/devices/dxl/dxl_driver_v1.py", line 85, in read_a_block
    vals = read_a_block_vals(port, idn, read_block, read_wait_time)
  File "/home/gautham/src/senseact/senseact/devices/dxl/dxl_driver_v1.py", line 115, in read_a_block_vals
    return read_block.vals_from_data(vals)
  File "/home/gautham/src/senseact/senseact/devices/dxl/dxl_reg.py", line 253, in vals_from_data
    return self.vals_from_dxl_data(data)
  File "/home/gautham/src/senseact/senseact/devices/dxl/dxl_reg.py", line 283, in vals_from_dxl_data
    vals.append(reg.unit.fwd(b))
  File "/home/gautham/src/senseact/senseact/devices/dxl/dxl_unit_conv.py", line 162, in fwd
    raise ValueError(x)
ValueError: 0
```

If you see this error message, it is highly likely that you are using the wrong baudrate or device ID number
to establish a serial connection with your DXL device.

## IOError
If you are unable to establish a serial connection with your DXL, try running this command and then try to
run your DXl test:

```bash
 sudo chmod a+rw /dev/ttyACM0
```


