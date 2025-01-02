# ORCA-WHISPER
ORCA-WHISPER: An Automatic Killer Whale Sound Type Generation Toolkit Using Deep Learning

## General Description
ORCA-WHISPER presents a deep bioacoustic signal generation framework, using generative
adversarial learning, trained on low-resource killer whale (Orcinus Orca) call type data. Besides audiovisual inspection, supervised call type classification, and model transferability to other acoustic domains, the auspicious quality of generated fake vocalizations was further demonstrated by visualizing, representing, and enhancing the real-world orca signal data manifold.

## Reference
If ORCA-WHISPER is used for your own research please cite the following publication: ORCA-WHISPER: An Automatic Killer Whale Sound Type Generation Toolkit Using Deep Learning:

```
@inproceedings{BerglerWHISPER:2022,
author={Christian Bergler and Alexander Barnhill and Dominik Perrin and Manuel Schmitt and Andreas Maier and Elmar Nöth},
title={{ORCA-WHISPER: An Automatic Killer Whale Sound Type Generation Toolkit Using Deep Learning}},
year=2022,
booktitle={Proc. Interspeech 2022},
pages={2413--2417},
doi={10.21437/Interspeech.2022-846}
}
```
## License
GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007 (GNU GPLv3)

# Installation
## LTFAT (Large Time Frequency Analysis Toolbox)
- It is recommended to install this package **globally** and to follow the installation instructions
provided by the devloper [Here](http://dev.pages.lis-lab.fr/ltfatpy/install.html)

# Configuration
The configuration of ORCA-WHISPER is controlled through the [parameters.json](./parameters.json) file in the root directory


# Training
- Set the desired parameters in [parameters.json](./parameters.json)
- Run [main.py](./main.py)

# Sound Sample Generation
- After you are satisfied with the result produced by the generator, initialize the `GANSampler` class in [generate.py](./generate.py)
```python
    parameter_file = "parameters.json"
    parameters = Parameters(parameter_file)
    g_inverter = GaborInverter(parameters.data, device=torch.device("cpu"))
    name_gen = OrcaSpotNameGenerator()
    checkpoint_path = "ORCA.pt"
    output_dir = "output"
    class_name = "ORCA"
    sample_count = 100
    noise_generator = GaussianNoiseGenerator(parameters=GaussianLatentSpaceParameters(),
                                                      batch_size=1,
                                                      latent_dimension_size=100)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    GANSampler(
            name_generator=name_gen,
            class_name=class_name,
            checkpoint_path=checkpoint_path,
            noise_generator=noise_generator,
            device=torch.device("cpu"),
            f_min=0,
            f_max=10000,
            inverter=g_inverter,
            output_base=output_dir)
    )(sample_count)
```