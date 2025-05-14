# bible-audiobook

## Why? Because I can, that's why

I grabbed a copy of the American King James Bible at [openbible.com](https://openbible.com/texts.htm)  
I wrote me a script to call the mx_audio.server and convert, verse by verse.
It took me **28 hours and 05 minutes** to convert all **31,102** verses, resulting in **86h32m36s**.
I ran it on a Mac Mini M4pro, 12 cores, 24GB Ram;

![Screenshot 2025-03-18 at 14 52 35](https://github.com/user-attachments/assets/4a213b39-a24c-4e93-b0dc-04b385ea20ed)


Some example results, are located at `./audios/bible_akjv`

## Want to try it yourself?

#### just do it!

#### Notes:

- I am a javascript developer so all the code is in javascript / typescript
- I work specifically with bun, and there is code here that won't work in nodejs
- for the code to work flawlessly (fault tolerant) and as is, you need to also have a nodejs installation so you can run the mlx_audio.server running through [pm2](https://pm2.keymetrics.io/)
  - you can do it without node and pm2 but there's a close to 100% chance the mlx_audio.server will crash a few times during the hard work so i built a mechanism that depends on pm2 so it doesn't need my manual intervention to keep going
- before running any of the scripts, edit the file to change your output directories and what not
- if the process does crash, or if you need to stop / pause the process, change the value on _line 37_ of `./src/index.ts` so you can pick up where it left off

### Instructions

install [Bun](https://bun.sh) if you don't have it

```bash
curl -fsSL https://bun.sh/install | bash
```

- Install local dependencies:

```bash
bun install
```

to synthesize the whole American King James Bible, run:

```bash
bun run src/index.ts
```

to convert all the .wav files to .mp3 (needs ffmpeg on your machine) run

```bash
bun run src/convert-to-mp3.ts
```

for statistics and to check the integrity of the **mp3** files (needs ffmpeg on your machine), run

```bash
bun run src/mp3-checker.ts
```

This project was created using `bun init` in bun v1.2.0. [Bun](https://bun.sh) is a fast all-in-one JavaScript runtime.
