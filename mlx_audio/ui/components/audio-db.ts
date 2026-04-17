const DB_NAME = "mlx-audio-db"
const DB_STORE = "audio-files"

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1)
    req.onupgradeneeded = () => req.result.createObjectStore(DB_STORE)
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

export async function saveAudioToDB(id: string, blob: Blob): Promise<void> {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(DB_STORE, "readwrite")
    tx.objectStore(DB_STORE).put(blob, id)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

export async function getAudioFromDB(id: string): Promise<string | null> {
  const db = await openDB()
  return new Promise((resolve) => {
    const req = db.transaction(DB_STORE).objectStore(DB_STORE).get(id)
    req.onsuccess = () => {
      if (!req.result) return resolve(null)
      resolve(URL.createObjectURL(req.result))
    }
    req.onerror = () => resolve(null)
  })
}

export async function deleteAudioFromDB(id: string): Promise<void> {
  const db = await openDB()
  return new Promise((resolve) => {
    const tx = db.transaction(DB_STORE, "readwrite")
    tx.objectStore(DB_STORE).delete(id)
    tx.oncomplete = () => resolve()
  })
}