### NIR DRP for LAM AIT

We are extending the existing PFS DRP and `drpActor` mechanisms to automatically generate "CCD-like" single images with masked pixels for each H4 ramp. This will be added to the DRP weeklies starting with `w.2022.31`.

Just as for CCD frames, the `drpActor` listens for ramp completion keywords from the `hxActor`. Those trigger a DRP ingest and simple ISR. In the case of H4 ramps, a `postISRCCD` file is generated, which in the current simple processing is the sum of the flux in the ramp: the last read minus the first read.

There are many parts of the DRP which will be considerably more sophisticated for science operations, but that file should suffice for engineering.

One step which will take ongoing work is the generation of the bad pixel mask. These detectors have a rich zoology of maskable pixels; for the 18315 device being shipped in SN1 we expect about 2.5% bad pixels, some evenly spread across the device, some associated with obvious detector defects, and some clustered near edges and corners.

The DRP currently only processes IRP 1:1 data -- see below for a description -- but by the time LAM takes data it will be able to process no-IRP data as well. The reduction routine for that is in use but has not yet been integrated into the DRP.

### NIR acquisition

H4 acquisition requires two MHS products: `hxhal` for low-level SAM and ASIC control and `ics_hxActor` for MHS actor control and file creation.

Ramp acquisition is driven by a single ASIC clock, which can usually be thought of as a line pointer which loops over the rows of the detector. For PFS this clock runs continuously, even when we are not taking data. At 100kpix/s and 32 readout channels each read takes 10.86s for IRP 1:1 and 5.45s for non-IRP.

The single MHS acquisition command is, for example, `hx_n3 ramp nread=5`, which maps pretty directly to a single primitive ASIC ramp command. We wait for the internal ASIC line pointer to carry back to the top of the detectors, after which we are fed a stream of pixels which we chop up into the individual reads. Each ramp is saved in a single PFSB file, with one HDU for each detector read, interleaved with one HDU for the corresponding reference read.

Because of the free-running line counter, the reset frame, and the need for differential pairs of data reads, the shortest useful ramp takes between three and four read times, averaging 38s for IRP 1:1 or 19s for no IRP. 

A decision we still need to make is whether to use the H4 Interleaved Reference Pixels ("IRP"). As it stands we acquire and process ramps with IRP 1:1, meaning that for each detector pixel we read a reference pixel from a dedicated row of H4RG reference pixels. Recent work by Erin Kado-Fong suggests that we will probably *not* use the IRP, and instead use the four border reference pixels found on all H2RG and H4RG detectors. This would shorten (halve!) per-read times.

### illumination times vs. multi-read ramps

Remember that reductions are always differential (or fit to 2+ running samples). For that to be sensible, we need either two reads sampling a constant illumination (i.e. by turning lamps on and off before and after we get two full reads), or three+ reads where the first read samples the detector before the illumination starts, and the last after the illumination has been turned off. 

When the reset read begins, the `hxActor` does learn and does propagate the absolute times expected for the start of the first data read. That could sometimes help, for example when using *very* bright sources (lit for fractions of a second), or when trying to synchronize with the shutter, but in general we want the first and last data frames to have stable illumination.

### MHS integration

We still need to write the top-level `iic` command to control the `hx ramp` commands in concert with the shutters and calibration lamps. As just described, the relative timings are not always trivial, especially for short illumination times. In particular, supporting the existing `iic expose exptime=SSS` is likely to be strange. Our current idea is to essentially round up, depending on the acquisition type (darks, DCB/pfiLamps, science), but we need to flesh out the details. Arnaud expects to be working on that later in August.


