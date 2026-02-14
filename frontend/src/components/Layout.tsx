import { NavLink, Outlet } from "react-router-dom";
import { Search, FolderOpen, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { to: "/", icon: Search, label: "Search" },
  { to: "/browse", icon: FolderOpen, label: "Browse" },
  { to: "/status", icon: Activity, label: "Status" },
];

export function Layout() {
  return (
    <div className="flex h-screen">
      <nav className="w-48 border-r border-border bg-card flex flex-col p-3 gap-1">
        <div className="text-lg font-bold px-3 py-2 mb-2 text-primary">QMD</div>
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors",
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              )
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
