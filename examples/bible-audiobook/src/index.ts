import { createReadStream } from "node:fs";
import path from 'node:path'
import readline from "node:readline";
import { rename } from "node:fs/promises";
import { mkdirp } from "mkdirp"
import { $ } from 'bun';
const regex = /(.*)\s(\d+\:\d+)\t(.*)/

const outputPath = '/path/to/output/dir';

const main = async (bibleName: string, voice: string, withTitle: boolean = false) => {
  const file = `./bibles/${bibleName}.txt`;
  await mkdirp(`./audios/${bibleName}`);

  const targetPath = path.resolve(__dirname, '..', 'audios', bibleName, voice);
  await mkdirp(targetPath);

  let lineCount = 0;
  // for (let voice of voices) {
  const rs = createReadStream(file);
  const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });
  await mkdirp(`./audios/${bibleName}/${voice}`);
  for await (const line of rl) {
    const match = line.match(regex);
    if (!match) continue;
    const [book, chapter, text] = match.slice(1);
    console.log(`${lineCount}: ${book} | ${chapter} | ${text}`);
    const fileName = `${leftpad((++lineCount).toString(), 8)}-${book}-${chapter!.replace('/', ':')}.wav`;
    const title = `${book} ${chapter}`;

    /**
     * in case we have to stop the process and resume later,
     * set the lineCount to the last index you processed
     * or better yet, a few indexes before, just to make sure
     */

    if (lineCount <= 0) continue;

    console.log(title, fileName)
    const remoteFileName = await transcribe(title, text!, voice);
    await rename(`${outputPath}/${remoteFileName}`, path.resolve(targetPath, fileName));

  }
};

main("bible-akjv", 'am_michael', false);

const transcribe = async (title: string, text: string, voice: string) => {
  try {
    const result = await fetch("http://localhost:3333/tts", {
      "headers": {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.8",
        // "content-type": "multipart/form-data; boundary=----WebKitFormBoundaryL6e2mv0qkI5aWBAs",
        "sec-ch-ua": "\"Not(A:Brand\";v=\"99\", \"Brave\";v=\"133\", \"Chromium\";v=\"133\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"macOS\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "sec-gpc": "1",
        "cookie": "NEXT_LOCALE=en; token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjIwZTdhMDFmLWIwMjItNDZjZi1iZjdhLWQ4ZmYwMzU1YWI4MSJ9.CblzSRvfQlUxVZGd08aVHpqbD7bRAJLON8XXpbV5Py0",
        "Referer": "http://localhost:3333/",
        "Referrer-Policy": "strict-origin-when-cross-origin"
      },
      "body": getBody(title, text, voice),
      "method": "POST"
    });
    const data: any = await result.json()
    return data.filename;
  } catch (ex) {
    // await fetch('https://ntfy.andrepadez.com/mlx-audio', {
    //   method: 'POST', // PUT works too
    //   body: 'Macmini bible synth process has crashed'
    // });

    /**
     * for this safeguard to work, we need to be running mlx-audio.server
     * through pm2, so it can restart itself when it crashes
     * and continue seemlessly from where it left off
     */

    console.log('ERRORED!', 'restarting mlx-audio server in 5 seconds');
    await new Promise((resolve) => setTimeout(resolve, 5000));
    await $`/opt/homebrew/bin/pm2 restart "MLX-AUDIO"`.text();
    await new Promise((resolve) => setTimeout(resolve, 5000));
    return transcribe(title, text, voice);
  }
}

const getBody = (title: string, text: string, voice: string) => {
  const formData = new FormData();
  formData.append("text", `\n\n${text}`);
  formData.append("voice", voice);
  formData.append("model", "mlx-community/Kokoro-82M-bf16");
  formData.append("speed", "1");
  return formData;
}




const voices = [
  "af_heart",
  "af_nova",
  "af_bella",
  "af_nicole",
  "af_sarah",
  "af_sky",
  "am_adam",
  "am_michael",
  "bf_emma",
  "bf_isabella",
  "bm_george",
  "bm_lewis",
]

const leftpad = (str: string, len: number = 2, ch: string = "0") => {
  const length = len - str.length;
  return ch.repeat(length) + str;
}