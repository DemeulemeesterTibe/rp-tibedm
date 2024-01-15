"use client"
import { redirect } from 'next/navigation'
import { DemoChat } from '@/components/component/demo-chat';

// Define the Home component
export default function Home() {
  redirect('/demo')
}
