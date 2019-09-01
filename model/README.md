# About Model

## Introduction

Most of the code is copy & change from the [github.com/openai/roboschoo](https://github.com/openai/roboschool), licence is MIT.

Thanks for your great work and open source spirit.

## Changes

We made several changes to custom the environment:

- Change the force limit to [-10,+10], for more flexible control
- Remove the `swingup` and we should set & calc reward by ourselves.
    - At the same time, we have to set initial position for ourselves.
- Of course, registered name changed to `TsingJyujingInvertedPendulum-v1` and `TsingJyujingInvertedDoublePendulum-v1`