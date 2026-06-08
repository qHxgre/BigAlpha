# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `alphathonapiserver`, a FastAPI-based competition platform API server built with BigQuant's shared framework (`bigshared2`). The application manages competition registration, scoring, and related functionality.

## Architecture

- **Framework**: Built on `BigAPIApp` from `bigshared2.bigapi`
- **Database**: Uses Tortoise ORM with Aerich for migrations
- **API Structure**: RESTful endpoints organized by domain (competition, user, leaderboard, submission, team)
- **Authentication**: Integrated with BigQuant's privilege system (`bigshared2.auth`)

### Key Components

- `src/alphathonapiserver/main.py`: Application entry point with BigAPIApp setup
- `src/alphathonapiserver/api/`: API routes organized by domain
  - `competitions.py`: Competition management endpoints
  - `users.py`: User registration endpoints
  - `leaderboard.py`: Leaderboard and scoring endpoints
  - `submissions.py`: Submission management endpoints
  - `teams.py`: Team management endpoints
- `src/alphathonapiserver/models/__init__.py`: Core data models using bigshared2 mixins
  - `Competition`: Competition entity with summary/data JSON fields
  - `User`: Competition-scoped user registrations with status tracking
  - `Submission`: User submissions with public/private scoring
  - `Team`: Team management with members and pending users
- `src/alphathonapiserver/constants.py`: Enums (UserStatus) and privilege definitions
- `src/alphathonapiserver/settings.py`: Database and environment configuration
- `migrations/`: Database migration files managed by Aerich

### Model Architecture

The models follow bigshared2 patterns using standardized mixins:
- `UUIDPrimaryKeyMixin`: UUID primary keys
- `CreatedAtMixin`/`UpdatedAtMixin`: Automatic timestamps
- `SpaceIDAndIndexMixin`: Multi-tenant support
- `CreatorAndIndexMixin`: Creator tracking

Models use JSON fields extensively for flexible data storage (summary, data, score_data fields).

## Development Commands

### Package Management
- Uses `uv` for dependency management
- Package index: `https://repos.bigquant.ai/artifactory/api/pypi/virtual.pypi/simple`

### Running the Application
```bash
# Development mode with debug enabled
task dev-serve

# Production mode
task prod

# Direct serve (production)
task serve

# Local development (alternative)
make dev_run
```

### Database Operations
```bash
# Initialize and upgrade database
task dbup

# Create new migration
make migrate
# or: uv run aerich migrate

# Apply migrations
make upgrade
# or: uv run aerich upgrade
```

### Code Quality
- **Linting**: Uses Ruff with strict configuration (line length: 160)
- **Type Checking**: MyPy configured for Python 3.11
- **Code Style**: Google docstring convention, comprehensive linting rules

```bash
# Lint code (configured in pyproject.toml)
ruff check .

# Type check
mypy src/
```

### Testing
- Test framework: Uses pytest (configured in build system)
- Test files located in `tests/` directory
- Run tests via the build system's pytest integration

## Configuration

### Environment Variables
- `DEBUG_AUTH=True`: Enable debug authentication mode
- `DEBUG=True`: Enable debug mode
- `API_DOCS_URL_HOST`: Host for API documentation
- `UVICORN_WORKERS`: Number of Uvicorn workers (default: 4)

### Database Configuration
- Configured via `TORTOISE_ORM` in settings.py
- Uses BigQuant's shared database settings (`bigshared2.db.sql.settings`)
- Models located in `alphathonapiserver.models`

## Build System
- Uses Nuitka for building
- Docker support with multi-stage builds
- Integrated with BigQuant's build toolchain in `public/build/`

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
