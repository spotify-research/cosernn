# cosernn

A reference implemententation of the CoSeRNN model for contextual music
recommendation, presented in the following paper:

> Casper Hansen, Christian Hansen, Lucas Maystre, Rishabh Mehrotra, Brian
> Brost, Federico Tomasi, Mounia Lalmas. _[Contextual and Sequential User
> Embeddings for Large-Scale Music
> Recommendation](https://dl.acm.org/doi/10.1145/3383313.3412248)_, RecSys
> 2020.


## Getting Started

Our implementation requires Python 3.7 and TensorFlow 1.x. To run the code, you
will need a CUDA-enabled GPU.

To get started, simply follow these steps:

- Clone the repo locally with: `git clone
  https://github.com/spotify-research/cosernn.git`
- Move to the repository with: `cd cosernn`
- install the dependencies: `pip install -r requirements.txt`
- install the package: `pip install -e lib/`

Generate data using

    python scripts/generate_data.py

Train the CoSeRNN model using

    python scripts/train.py path/to/records


## Support

Create a [new issue](https://github.com/spotify-research/cosernn/issues/new)


## Contributing

We feel that a welcoming community is important and we ask that you follow Spotify's
[Open Source Code of Conduct](https://github.com/spotify/code-of-conduct/blob/master/code-of-conduct.md)
in all interactions with the community.


## Authors

- [Casper Hansen](mailto:casper.hanzen@gmail.com)
- [Lucas Maystre](mailto:lucasm@spotify.com)

A full list of [contributors](https://github.com/spotify-research/cosernn/graphs/contributors?type=a) can
be found on GitHub.

Follow [@SpotifyResearch](https://twitter.com/SpotifyResearch) on Twitter for
updates.


## License

Copyright 2020 Spotify, Inc.

Licensed under the Apache License, Version 2.0:
https://www.apache.org/licenses/LICENSE-2.0


## Security Issues?

Please report sensitive security issues via Spotify's bug-bounty program
(https://hackerone.com/spotify) rather than GitHub.
