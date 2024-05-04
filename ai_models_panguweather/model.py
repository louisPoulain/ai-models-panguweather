# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import yaml

import numpy as np
import onnxruntime as ort
from ai_models.model import Model
import xarray as xr

LOG = logging.getLogger(__name__)


class PanguWeather(Model):
    # Download
    download_url = (
        "https://get.ecmwf.int/repository/test-data/ai-models/pangu-weather/{file}"
    )
    download_files = ["pangu_weather_24.onnx", "pangu_weather_6.onnx"]

    # Input
    area = [90, 0, -90, 360]
    grid = [0.25, 0.25]
    param_sfc = ["msl", "10u", "10v", "2t"]
    param_level_pl = (
        ["z", "q", "t", "u", "v"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    )

    # Output
    expver = "pguw"

    def __init__(self, num_threads=1, **kwargs):
        super().__init__(**kwargs)
        self.num_threads = num_threads

    def run(self):
        fields_pl = self.fields_pl

        param, level = self.param_level_pl
        
        if isinstance(self.all_fields, list):
            self.param_sfc = ["msl", "u10", "v10", "t2m"]
            fields_pl = fields_pl.sel(isobaricInhPa=level)[param]
            fields_sfc = self.fields_sfc[self.param_sfc]
            
            fields_pl_numpy = np.concatenate([fields_pl[f].values for f in param])
            fields_sfc_numpy = np.concatenate([fields_sfc[f].values for f in self.param_sfc])
            
        else:
            fields_pl = fields_pl.sel(param=param, level=level)
            fields_pl = fields_pl.order_by(param=param, level=level)

            fields_pl_numpy = fields_pl.to_numpy(dtype=np.float32)
            fields_pl_numpy = fields_pl_numpy.reshape((5, 13, 721, 1440))

            fields_sfc = self.fields_sfc
            fields_sfc = fields_sfc.sel(param=self.param_sfc)
            fields_sfc = fields_sfc.order_by(param=self.param_sfc)

            fields_sfc_numpy = fields_sfc.to_numpy(dtype=np.float32)

        input = fields_pl_numpy
        input_surface = fields_sfc_numpy

        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = self.num_threads

        pangu_weather_24 = os.path.join(self.assets, "pangu_weather_24.onnx")
        pangu_weather_6 = os.path.join(self.assets, "pangu_weather_6.onnx")

        # That will trigger a FileNotFoundError

        os.stat(pangu_weather_24)
        os.stat(pangu_weather_6)

        with self.timer(f"Loading {pangu_weather_24}"):
            ort_session_24 = ort.InferenceSession(
                pangu_weather_24,
                sess_options=options,
                providers=self.providers,
            )

        with self.timer(f"Loading {pangu_weather_6}"):
            ort_session_6 = ort.InferenceSession(
                pangu_weather_6,
                sess_options=options,
                providers=self.providers,
            )

        input_24, input_surface_24 = input, input_surface
        
        if isinstance(self.all_fields, list):
            data_vars = {}
            surface_outputs = []
            pl_outputs = []

        with self.stepper(6) as stepper:
            for i in range(self.lead_time // 6):
                step = (i + 1) * 6

                if (i + 1) % 4 == 0:
                    output, output_surface = ort_session_24.run(
                        None,
                        {
                            "input": input_24,
                            "input_surface": input_surface_24,
                        },
                    )
                    input_24, input_surface_24 = output, output_surface
                else:
                    output, output_surface = ort_session_6.run(
                        None,
                        {
                            "input": input,
                            "input_surface": input_surface,
                        },
                    )
                input, input_surface = output, output_surface

                # Save the results

                if not isinstance(self.all_fields, list):
                    pl_data = output.reshape((-1, 721, 1440))
                    for data, f in zip(pl_data, fields_pl):
                        self.write(data, template=f, step=step)

                    sfc_data = output_surface.reshape((-1, 721, 1440))
                    for data, f in zip(sfc_data, fields_sfc):
                        self.write(data, template=f, step=step)
                else:
                    surface_outputs.append(output_surface)
                    pl_outputs.append(output)
                
                stepper(i, step)
            
            if isinstance(self.all_fields, list):
                surface_output = np.stack(surface_outputs).transpose((1, 0, 2, 3))
                pl_output = np.stack(pl_outputs).transpose((1, 0, 2, 3, 4))
                for i in range(len(self.param_sfc)):
                    data_vars[self.param_sfc[i]] = (
                        ("time", "lat", "lon"),
                        surface_output[i],
                    )
                for i in range(len(self.param_level_pl[0])):
                    data_vars[self.param_level_pl[0][i]] = (
                        ("time", "level", "lat", "lon"),
                        pl_output[i],
                    )
                
                steps = np.arange(6, self.lead_time + 6, 6)    
                times = [self.all_fields[1].time.values[0] + np.timedelta64(steps[i], 'h') for i in range(len(steps))]
                lat, lon = self.all_fields[0].latitude.values[::-1], self.all_fields[0].longitude.values
                saved_xarray = xr.Dataset(
                    data_vars=data_vars,
                    coords=dict(
                        lon=lon,
                        lat=lat,
                        time=times,
                        level=self.param_level_pl[1],
                    ),
                )
                saved_xarray = saved_xarray.reindex(level=saved_xarray.level[::-1])
                saved_xarray = saved_xarray.rename({"level": "isobaricInhPa"})
                start_date = self.all_fields[0].valid_time.values[0]
                
                with open("/users/lpoulain/louis/TCBench_0.1/slurms/models_config.yml", 'r') as f:
                    folder = yaml.full_load(f).get("pangu_folder")
                name = f"pangu_{np.datetime64(start_date, 'h')}_to_{np.datetime64(start_date + np.timedelta64(self.lead_time, 'h'), 'h')}"+\
                    f"_ldt_{self.lead_time}.nc"
                name = os.path.join(folder, name)
                    
                LOG.info(f"Saving to {name}")
                #encoding = {}
                #for data_var in output.data_vars:
                #    encoding[data_var] = {
                #    "original_shape": output[data_var].shape,
                #    "_FillValue": -32767,
                #    "dtype": np.int16,
                #    "add_offset": output[data_var].mean().compute().values,
                #    "scale_factor": output[data_var].std().compute().values / 1000, # save up to 32 std
                #    # "zlib": True,
                #    # "complevel": 5,
                #    }
                #output.to_netcdf(name, engine="netcdf4", mode="w", encoding=encoding, compute=True)
                saved_xarray.to_netcdf(name, engine="netcdf4", mode="w", compute=True)
