import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { saveBase64Image } from '@/lib/utils/image-storage'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

async function findActiveClass(timestamp: string, room: string) {
  const currentDate = new Date(timestamp)
  const utcTimestamp = currentDate.toISOString()
  
  const { data: activeClass, error } = await supabase
    .from('classes')
    .select('id, end_time')
    .eq('room', room)
    .lte('start_time', utcTimestamp)
    .gte('end_time', utcTimestamp)
    .single()

  if (error) {
    console.error('Error finding active class:', error)
    return null
  }

  return activeClass
}

export async function POST(request: Request) {
  try {
    const { records } = await request.json()
    
    if (!Array.isArray(records) || records.length === 0) {
      return NextResponse.json(
        { error: 'Invalid records format' },
        { status: 400 }
      )
    }

    const activeClass = await findActiveClass(records[0].timestamp, records[0].room)
    
    if (!activeClass) {
      throw new Error(`No active class found for room ${records[0].room} at ${records[0].timestamp}`)
    }

    const processedRecords = await Promise.all(
      records.map(async (record) => {
        const utcTimestamp = new Date(record.timestamp).toISOString()
        const imagePath = await saveBase64Image(
          record.image,
          record.name,
          utcTimestamp.replace(/[:.]/g, '-')
        )

        return {
          student_name: record.name,
          confidence: record.confidence,
          quality: record.quality,
          timestamp: utcTimestamp,
          room: record.room,
          image_path: imagePath,
          class_id: activeClass.id
        }
      })
    )

    const { data, error } = await supabase
      .from('attendance')
      .insert(processedRecords)

    if (error) {
      console.error('Error storing records:', error)
      return NextResponse.json(
        { error: 'Failed to store records' },
        { status: 500 }
      )
    }

    return NextResponse.json({ 
      success: true, 
      stored: processedRecords.length,
      classEndTime: activeClass.end_time
    })
    
  } catch (error) {
    console.error('Error processing request:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
} 