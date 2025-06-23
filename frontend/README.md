# Finance RAG System - Frontend

A modern, responsive web interface for the Finance RAG (Retrieval-Augmented Generation) System, built with React, TypeScript, and Material-UI.

## Features

- 📂 Upload and manage financial documents (PDF, DOCX, TXT, CSV, XLSX)
- 💬 Chat interface for querying documents
- 🎨 Modern, responsive UI with dark/light mode support
- 📊 View document sources and references
- 🔍 Full-text search across uploaded documents
- 🚀 Optimized for performance and user experience

## Prerequisites

- Node.js (v16 or later)
- npm (v7 or later) or yarn
- Backend API (see backend setup instructions)

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Configure environment variables**
   Create a `.env` file in the root directory with the following variables:
   ```env
   REACT_APP_API_BASE_URL=http://localhost:8080
   ```

4. **Start the development server**
   ```bash
   npm start
   # or
   yarn start
   ```

   The application will be available at `http://localhost:3000`

## Available Scripts

- `npm start` - Start the development server
- `npm test` - Run tests
- `npm run build` - Build the application for production
- `npm run eject` - Eject from create-react-app (use with caution)
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier

## Project Structure

```
frontend/
├── public/                 # Static files
├── src/
│   ├── components/        # Reusable UI components
│   ├── hooks/              # Custom React hooks
│   ├── pages/              # Page components
│   ├── services/           # API services
│   ├── theme/              # Theme configuration
│   ├── types/              # TypeScript type definitions
│   ├── utils/              # Utility functions
│   ├── App.tsx             # Main application component
│   └── index.tsx           # Application entry point
├── .env                    # Environment variables
├── package.json            # Project dependencies and scripts
└── tsconfig.json           # TypeScript configuration
```

## Styling

This project uses:

- [Material-UI](https://mui.com/) - React UI component library
- [Emotion](https://emotion.sh/) - CSS-in-JS solution
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework

## State Management

- [React Query](https://tanstack.com/query) - Data fetching and state management
- React Context API - For global state

## API Integration

The frontend communicates with the backend REST API. See the API documentation for available endpoints.

## Testing

```bash
# Run tests
npm test

# Run tests in watch mode
npm test -- --watch

# Generate test coverage report
npm test -- --coverage
```

## Building for Production

```bash
# Create a production build
npm run build

# Serve the production build locally
npx serve -s build
```

## Deployment

The application can be deployed to any static hosting service:

- [Vercel](https://vercel.com/)
- [Netlify](https://www.netlify.com/)
- [GitHub Pages](https://pages.github.com/)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Create React App](https://create-react-app.dev/)
- [Material-UI](https://mui.com/)
- [React Query](https://tanstack.com/query)
- [TypeScript](https://www.typescriptlang.org/)
