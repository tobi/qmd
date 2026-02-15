import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { SearchPage } from "./pages/SearchPage";
import { DocumentPage } from "./pages/DocumentPage";
import { BrowsePage } from "./pages/BrowsePage";
import { StatusPage } from "./pages/StatusPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<SearchPage />} />
        <Route path="/doc/*" element={<DocumentPage />} />
        <Route path="/browse" element={<BrowsePage />} />
        <Route path="/browse/:collection" element={<BrowsePage />} />
        <Route path="/browse/:collection/*" element={<BrowsePage />} />
        <Route path="/status" element={<StatusPage />} />
      </Route>
    </Routes>
  );
}
