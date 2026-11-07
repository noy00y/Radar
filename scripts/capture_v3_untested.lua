-- capture.lua
-- Capture only. Same steps as capture_v2.lua (which worked):
-- arm DCA1000 -> start frames -> wait -> stop -> check file size.
-- The radar profile/chirp/frame settings must already be loaded in mmWave Studio.
--
-- Before each run, edit the 5 settings below. Values for each experiment:
--
--   experiment           config to load in Studio               frames  capture_ms  expected_bytes
--   calibration overlook radar_capture_highway_overlook.yaml        64       6000        31457280
--   calibration car reg. radar_capture_vehicle.yaml                 64       9000         8388608
--   calibration car fast radar_capture_vehicle_highway.yaml         64       4000         3932160
--   overlook (Exp 2)     radar_capture_highway_overlook.yaml       600      35000       294912000
--   car regular (Exp 3)  radar_capture_vehicle.yaml                128      16000        16777216
--   car fast (Exp 3)     radar_capture_vehicle_highway.yaml       3000      65000       184320000
--
-- Calibration: name = "cal_0deg" (reflector straight ahead), then "cal_27deg" (off to one side),
-- using the SAME config as the experiment you are calibrating for.

local output_dir = "C:\\Users\\ozair\\Documents\\Code_Drive\\Drone\\radar\\claude_v1\\bins"

local name = "overlook"            -- file name prefix
local capture_count = 1            -- how many captures to record back to back
local capture_ms = 35000           -- wait for the frames to finish (frames x frame period + margin)
local expected_bytes = 294912000   -- from the table above

local function file_size(path)
    local f = io.open(path, "rb")
    if not f then
        return -1
    end

    local size = f:seek("end")
    f:close()
    return size or -1
end

for i = 1, capture_count do
    local output_path = string.format(
        "%s\\%s_%03d.bin",
        output_dir,
        name,
        i
    )

    WriteToLog("\n[STAP] Arming capture " .. i .. ": " .. output_path .. "\n", "green")

    ar1.CaptureCardConfig_StartRecord(output_path, 1)
    RSTD.Sleep(2000)

    WriteToLog("[STAP] Starting frame...\n", "green")
    ar1.StartFrame()

    -- Allow the finite frame sequence to finish and data to drain.
    RSTD.Sleep(capture_ms)

    WriteToLog("[STAP] Stopping recording...\n", "green")
    ar1.CaptureCardConfig_StopRecord()
    RSTD.Sleep(1500)

    local size = file_size(output_path)

    if size == expected_bytes then
        WriteToLog("[STAP] Capture " .. i .. " size matches expected: " .. size .. " bytes\n", "green")
    elseif size >= 0 then
        WriteToLog(
            "[STAP] Capture " .. i .. " size was " .. size ..
            " bytes; expected " .. expected_bytes .. "\n",
            "red"
        )
    else
        WriteToLog(
            "[STAP] Could not check the output file. " ..
            "Inspect the actual DCA1000 output files.\n",
            "red"
        )
    end

    RSTD.Sleep(1000)
end

WriteToLog("[STAP] Capture loop finished.\n", "green")
