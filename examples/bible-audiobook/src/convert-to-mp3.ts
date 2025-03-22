import fs from 'node:fs/promises'
import path from 'node:path'
import { mkdirp } from 'mkdirp'
import ffmpeg from 'fluent-ffmpeg'

const origin = 'audios/bible-akjv/am_michael';
const destination = 'audios/bible-akjv/destination';

const regex = /(\d{8})-(.+)(-)(\d+)\:(\d+)(\.wav)/

const allFiles = (await fs.readdir(origin)).toSorted();

for (let file of allFiles) {
  // console.log(file)
  if (file.startsWith('.')) continue;
  const filePath = path.join(origin, file);
  const fileStat = await fs.stat(filePath);
  if (fileStat.isFile()) {
    const match = file.match(regex);
    //console.log(match);
    const [, index, book, , chapter, verse, extension] = match!;
    const idx = parseInt(index!, 10);

    // if (idx < 17850) continue;

    const destinationPath = path.join(destination, book!, chapter!);
    await mkdirp(destinationPath);
    const newFileName = `${index}_${book}_${chapter}_${verse}.mp3`;
    // console.log({ newFileName });
    const newFilePath = path.join(destinationPath, newFileName);
    console.log(idx, newFilePath)
    await new Promise((resolve, reject) => {
      ffmpeg(filePath)
        .toFormat('mp3')
        .on('end', resolve)
        .on('error', reject)
        .save(newFilePath);
    });
  } else {
    throw new Error('Not a file');
  }
  await new Promise(resolve => setTimeout(resolve, 500));
}