"use client"

import { Phone, BarChart3, Home } from "lucide-react"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
} from "@/components/ui/sidebar"
import Link from "next/link"
import Image from "next/image"
import logo from "../app/download.png"
import { usePathname } from "next/navigation"

const items = [
  {
    title: "कॉल शुरू करें",
    url: "/",
    icon: Phone,
  },
  {
    title: "डैशबोर्ड",
    url: "/dashboard",
    icon: BarChart3,
  },
]

export function AppSidebar() {
  const pathname = usePathname()

  return (
    <Sidebar>
      <SidebarHeader className="border-b">
        <div>
          <div className="flex items-center gap-2 px-4 py-2">
            <Image src={logo} alt="Logo" className="h-20 w-20" width={16} height={16} />
            {/* <logo className="h-4 w-4" /> */}
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold">PM-KUSUM</span>
            <span className="text-xs text-muted-foreground">Voice Agent</span>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>मुख्य मेनू</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild isActive={pathname === item.url}>
                    <Link href={item.url}>
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="border-t">
        <div className="p-4 text-xs text-muted-foreground">© 2024 PM-KUSUM Voice Agent</div>
      </SidebarFooter>
    </Sidebar>
  )
}
