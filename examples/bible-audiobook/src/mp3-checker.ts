const ffmpeg = require('fluent-ffmpeg');
const fs = require('fs').promises;
const path = require('path');

// Directory containing your MP3 files (root folder)
const rootDirectoryPath = '/path/to/output/dir'; // Replace with your root directory path

// Recursive function to get all MP3 files in a directory and its subdirectories
async function getAllMp3Files(dirPath: string) {
  let mp3Files: any = [];
  const files = await fs.readdir(dirPath, { withFileTypes: true });

  for (const file of files) {
    const fullPath = path.join(dirPath, file.name);
    if (file.isDirectory()) {
      // Recursively get MP3s from subdirectories
      const subDirFiles = await getAllMp3Files(fullPath);
      mp3Files = mp3Files.concat(subDirFiles);
    } else if (file.name.endsWith('.mp3') && !file.name.startsWith('.')) {
      // Add MP3 file with its full relative path
      mp3Files.push(fullPath);
    }
  }

  return mp3Files;
}

let counter: number = 1;

async function getMp3DurationsWithValidation() {
  try {
    // Get all MP3 files recursively
    const mp3Files = await getAllMp3Files(rootDirectoryPath);

    if (mp3Files.length === 0) {
      console.log('No MP3 files found in the directory or its subdirectories.');
      return;
    }

    let totalDuration = 0;
    let corruptFiles = 0;
    const corruptFileList = [];

    // Process each MP3 file
    for (const filePath of mp3Files) {
      // Use relative path from root for cleaner output
      const relativeFilePath = path.relative(rootDirectoryPath, filePath);

      try {
        const duration: number = await new Promise((resolve, reject) => {
          ffmpeg.ffprobe(filePath, (err: any, metadata: any) => {
            if (err) {
              return reject(new Error(`ffprobe error: ${err.message}`));
            }

            const durationInSeconds = metadata.format.duration;
            if (typeof durationInSeconds !== 'number' || isNaN(durationInSeconds) || durationInSeconds <= 0) {
              return reject(new Error('Invalid or missing duration in metadata'));
            }

            resolve(durationInSeconds);
          });
        });

        console.log(counter, `${relativeFilePath}: ${duration.toFixed(2)} seconds (Valid)`);
        totalDuration += duration;

      } catch (error: any) {
        console.log(counter, `${relativeFilePath}: CORRUPT or UNPARSEABLE - ${error.message}`);
        corruptFiles++;
        corruptFileList.push(`${relativeFilePath} - ${error.message}`);
      } finally {
        counter++;
      }
    }

    // Summary
    console.log(`\nProcessed ${mp3Files.length} MP3 files:`);
    console.log(`- Valid files: ${mp3Files.length - corruptFiles}`);
    console.log(`- Corrupt or unparseable files: ${corruptFiles}`);

    if (corruptFiles > 0) {
      console.log('\nList of corrupt or unparseable files:');
      corruptFileList.forEach(file => console.log(`- ${file}`));
    } else {
      console.log('\nNo corrupt or unparseable files found.');
    }

    if (totalDuration > 0) {
      // Convert total duration to human-readable format
      const hours = Math.floor(totalDuration / 3600);
      const minutes = Math.floor((totalDuration % 3600) / 60);
      const seconds = Math.floor(totalDuration % 60);

      console.log(`\nTotal duration of valid files: ${hours}h ${minutes}m ${seconds}s`);
      console.log(`Total duration in seconds: ${totalDuration.toFixed(2)}`);
    } else {
      console.log('\nNo valid files to calculate total duration.');
    }

  } catch (error) {
    console.error('An unexpected error occurred:', error);
  }
}

// Run the function
getMp3DurationsWithValidation();